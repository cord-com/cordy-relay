"""Cordy relay — connects Slack to Cordy's brain on Claude Managed Agents.

Pattern follows Anthropic's official slack_data_bot cookbook:
  mention -> create hosted session -> stream events back to the thread;
  thread replies continue the same session (no re-mention needed).

This process is a dumb pipe: all agent execution happens on Anthropic's
cloud. It only needs a WebSocket to Slack and HTTPS to Anthropic.
"""
import os
import re
import sqlite3
import threading

from anthropic import Anthropic
from dotenv import load_dotenv
from markdown_to_mrkdwn import SlackMarkdownConverter
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

client = Anthropic()
app = App(token=os.environ["SLACK_BOT_TOKEN"])
mrkdwn = SlackMarkdownConverter()

AGENT = {
    "type": "agent",
    "id": os.environ["CORDY_AGENT_ID"],
    "version": int(os.environ["CORDY_AGENT_VERSION"]),
}
ENV_ID = os.environ["CORDY_ENV_ID"]
VAULT_ID = os.environ.get("CORDY_VAULT_ID")

# thread -> session map, persisted so conversations survive restarts.
# Railway: attach a volume at /data so this file persists across deploys.
DB_PATH = os.environ.get("CORDY_DB", "/data/cordy_sessions.db"
                          if os.path.isdir("/data") else "cordy_sessions.db")
_db_lock = threading.Lock()
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS sessions"
           " (thread_key TEXT PRIMARY KEY, session_id TEXT NOT NULL)")
db.commit()


def get_session(channel: str, thread_ts: str):
    with _db_lock:
        row = db.execute("SELECT session_id FROM sessions WHERE thread_key=?",
                         (f"{channel}:{thread_ts}",)).fetchone()
    return row[0] if row else None


def save_session(channel: str, thread_ts: str, session_id: str) -> None:
    with _db_lock:
        db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)",
                   (f"{channel}:{thread_ts}", session_id))
        db.commit()


def post(channel: str, thread_ts: str, text: str) -> None:
    text = mrkdwn.convert(text)
    for i in range(0, max(len(text), 1), 3500):
        app.client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                    text=text[i:i + 3500])


def relay_stream(session_id: str, channel: str, thread_ts: str) -> None:
    """Read the hosted session's event stream; post the answer to Slack."""
    summary = ""
    for ev in client.beta.sessions.events.stream(session_id):
        t = ev.type
        if t == "agent.message":
            for b in ev.content:
                if b.type == "text" and b.text.strip():
                    summary = b.text
        elif t == "session.status_idle":
            break
        elif t == "session.status_terminated":
            post(channel, thread_ts,
                 "My session ended unexpectedly — Ben can check the trace at "
                 f"https://platform.claude.com/sessions/{session_id}")
            return
    post(channel, thread_ts, summary or "Done — nothing to report.")

    # Upload any files Cordy produced (reports etc.) to the thread.
    try:
        outputs = client.beta.files.list(scope_id=session_id,
                                         betas=["managed-agents-2026-04-01"])
        for f in outputs.data:
            if getattr(f, "downloadable", False):
                content = client.beta.files.download(f.id).read()
                app.client.files_upload_v2(channel=channel, thread_ts=thread_ts,
                                           filename=f.filename, content=content)
    except Exception:
        pass  # file delivery is best-effort


def run_turn(channel: str, thread_ts: str, user: str, text: str) -> None:
    try:
        session_id = get_session(channel, thread_ts)
        if session_id is None:
            session = client.beta.sessions.create(
                environment_id=ENV_ID,
                agent=AGENT,
                **({"vault_ids": [VAULT_ID]} if VAULT_ID else {}),
                title="".join(c for c in text if c.isprintable())[:80] or "Cordy",
                metadata={"slack_channel": channel, "slack_thread_ts": thread_ts},
            )
            session_id = session.id
            save_session(channel, thread_ts, session_id)
        client.beta.sessions.events.send(
            session_id,
            events=[{"type": "user.message",
                     "content": [{"type": "text",
                                  "text": f"<@{user}> asks: {text}"}]}],
        )
        relay_stream(session_id, channel, thread_ts)
    except Exception as e:
        post(channel, thread_ts,
             f"Something broke on my end ({type(e).__name__}). "
             "Ben, check the Railway logs and the Console.")


@app.event("app_mention")
def on_mention(event, ack, context):
    ack()
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    user = event.get("user", "someone")
    text = re.sub(rf"<@{context['bot_user_id']}>", "",
                  event.get("text", "")).strip()
    if not text:
        post(channel, thread_ts, f"Hi <@{user}> — what can I do for you?")
        return
    app.client.reactions_add(channel=channel, name="eyes",
                             timestamp=event["ts"])
    threading.Thread(target=run_turn,
                     args=(channel, thread_ts, user, text)).start()


@app.event("message")
def on_message(event, ack):
    ack()
    if event.get("subtype") or event.get("bot_id"):
        return
    # Public-only: DMs get a polite redirect (the River rule).
    if event.get("channel_type") == "im":
        app.client.chat_postMessage(
            channel=event["channel"],
            text="I work in the open so everyone learns from the answers — "
                 "mention me in a public channel like #cordy and I'll help "
                 "you there. :sunflower:")
        return
    # Thread replies continue an existing session without a re-mention.
    thread_ts = event.get("thread_ts")
    if thread_ts and get_session(event["channel"], thread_ts):
        threading.Thread(
            target=run_turn,
            args=(event["channel"], thread_ts,
                  event.get("user", "someone"), event.get("text", "")),
        ).start()


if __name__ == "__main__":
    print("Cordy relay online.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
