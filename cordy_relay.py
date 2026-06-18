"""Cordy relay — connects Slack to Cordy's brain on Claude Managed Agents.

Pattern follows Anthropic's official slack_data_bot cookbook:
  mention -> create hosted session -> stream events back to the thread;
  thread replies continue the same session (no re-mention needed).

This process is a dumb pipe: all agent execution happens on Anthropic's
cloud. It only needs a WebSocket to Slack and HTTPS to Anthropic.
"""
import os
import re
import base64
import sqlite3
import threading

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from markdown_to_mrkdwn import SlackMarkdownConverter
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

client = Anthropic()
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
app = App(token=SLACK_BOT_TOKEN)

# Image types Claude accepts. Slack files carry a mimetype we can check.
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGES_PER_TURN = 5            # keep sessions sane
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Claude's per-image ceiling

# Context caps (the grep-paper lesson: more text is not always better).
MAX_THREAD_MESSAGES = 50    # Layer 1: how much of the current thread to include
MAX_CHANNEL_MESSAGES = 25   # Layer 2: ambient channel history for top-level pings
MAX_CONTEXT_IMAGES = 5      # cap images pulled from surrounding context

mrkdwn = SlackMarkdownConverter()

# Cache user id -> display name so transcripts read "Sorrel:" not "U123:".
_user_cache: dict[str, str] = {}
_user_lock = threading.Lock()


def user_name(user_id: str) -> str:
    if not user_id:
        return "someone"
    with _user_lock:
        if user_id in _user_cache:
            return _user_cache[user_id]
    try:
        info = app.client.users_info(user=user_id)["user"]
        name = (info.get("profile", {}).get("display_name")
                or info.get("real_name") or user_id)
    except Exception:
        name = user_id
    with _user_lock:
        _user_cache[user_id] = name
    return name

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


def extract_images(event: dict) -> list[dict]:
    """Download any image files attached to a Slack event and return them as
    Claude image content blocks. Slack files sit behind an authenticated URL,
    so we fetch with the bot token. Needs the files:read scope.
    """
    blocks = []
    for f in event.get("files", []) or []:
        if len(blocks) >= MAX_IMAGES_PER_TURN:
            break
        mimetype = f.get("mimetype", "")
        if mimetype not in SUPPORTED_IMAGE_TYPES:
            continue  # skip PDFs, docs, etc. - images only for now
        url = f.get("url_private_download") or f.get("url_private")
        if not url:
            continue
        try:
            resp = requests.get(
                url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.content
            if len(data) > MAX_IMAGE_BYTES:
                continue  # too big for Claude; skip rather than fail the turn
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetype,
                    "data": base64.standard_b64encode(data).decode("ascii"),
                },
            })
        except Exception:
            continue  # one bad file shouldn't sink the whole message
    return blocks


def _format_messages(messages: list[dict], bot_user_id: str) -> tuple[str, list[dict]]:
    """Turn a list of Slack messages into a readable transcript + any images.
    Skips Cordy's own past replies (she already knows what she said) and the
    triggering message (added separately as the actual question).
    """
    lines = []
    images = []
    for m in messages:
        if m.get("bot_id") or m.get("user") == bot_user_id:
            continue  # skip Cordy's own messages
        txt = re.sub(rf"<@{bot_user_id}>", "@Cordy", m.get("text", "")).strip()
        name = user_name(m.get("user", ""))
        if txt:
            lines.append(f"{name}: {txt}")
        if len(images) < MAX_CONTEXT_IMAGES:
            images.extend(extract_images(m)[:MAX_CONTEXT_IMAGES - len(images)])
    return "\n".join(lines), images


def fetch_thread_context(channel: str, thread_ts: str,
                         bot_user_id: str) -> tuple[str, list[dict]]:
    """Layer 1: the full current thread (messages + files)."""
    try:
        resp = app.client.conversations_replies(
            channel=channel, ts=thread_ts, limit=MAX_THREAD_MESSAGES)
        text, imgs = _format_messages(resp.get("messages", []), bot_user_id)
        print(f"[context] thread fetch ok: {len(resp.get('messages', []))} msgs, "
              f"{len(text)} chars, {len(imgs)} images")
        return text, imgs
    except Exception as e:
        print(f"[context] thread fetch FAILED: {type(e).__name__}: {e}")
        return "", []


def fetch_channel_context(channel: str, bot_user_id: str) -> tuple[str, list[dict]]:
    """Layer 2: recent channel history, capped (avoids context rot)."""
    try:
        resp = app.client.conversations_history(
            channel=channel, limit=MAX_CHANNEL_MESSAGES)
        msgs = list(reversed(resp.get("messages", [])))
        text, imgs = _format_messages(msgs, bot_user_id)
        print(f"[context] channel fetch ok: {len(msgs)} msgs, "
              f"{len(text)} chars, {len(imgs)} images")
        return text, imgs
    except Exception as e:
        print(f"[context] channel fetch FAILED: {type(e).__name__}: {e}")
        return "", []


def run_turn(channel: str, thread_ts: str, user: str, text: str,
             images: list[dict] | None = None,
             is_thread_reply: bool = False, bot_user_id: str = "") -> None:
    try:
        session_id = get_session(channel, thread_ts)
        is_new_session = session_id is None

        context_text = ""
        context_images: list[dict] = []
        if is_new_session:
            # Only gather surrounding context when the conversation starts;
            # once a session exists it already remembers the running thread.
            if is_thread_reply:
                # Pinged inside an existing thread -> include that thread.
                context_text, context_images = fetch_thread_context(
                    channel, thread_ts, bot_user_id)
                label = "Here is the Slack thread so far, for context:"
            else:
                # Top-level ping -> include recent channel history (capped).
                context_text, context_images = fetch_channel_context(
                    channel, bot_user_id)
                label = ("Here is recent activity in this Slack channel, "
                         "for context:")

        if is_new_session:
            session = client.beta.sessions.create(
                environment_id=ENV_ID,
                agent=AGENT,
                **({"vault_ids": [VAULT_ID]} if VAULT_ID else {}),
                title="".join(c for c in text if c.isprintable())[:80] or "Cordy",
                metadata={"slack_channel": channel, "slack_thread_ts": thread_ts},
            )
            session_id = session.id
            save_session(channel, thread_ts, session_id)

        # Assemble the message: context block (new sessions only), then the
        # actual question, then any images attached to the triggering message.
        content = []
        if context_text:
            content.append({"type": "text",
                            "text": f"{label}\n\n{context_text}\n\n---"})
        if context_images:
            content.extend(context_images)
        content.append({"type": "text", "text": f"{user_name(user)} asks: {text}"})
        if images:
            content.extend(images)

        client.beta.sessions.events.send(
            session_id,
            events=[{"type": "user.message", "content": content}],
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
    images = extract_images(event)
    if not text and not images:
        post(channel, thread_ts, f"Hi <@{user}> — what can I do for you?")
        return
    if not text and images:
        text = "(see attached image(s))"
    # If the mention is inside an existing thread, pull the thread; otherwise
    # it's a top-level ping, so pull recent channel history.
    is_thread_reply = bool(event.get("thread_ts"))
    app.client.reactions_add(channel=channel, name="eyes",
                             timestamp=event["ts"])
    threading.Thread(
        target=run_turn,
        args=(channel, thread_ts, user, text, images,
              is_thread_reply, context["bot_user_id"]),
    ).start()


@app.event("message")
def on_message(event, ack, context):
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
        images = extract_images(event)
        threading.Thread(
            target=run_turn,
            args=(event["channel"], thread_ts,
                  event.get("user", "someone"),
                  event.get("text", "") or "(see attached image(s))", images,
                  True, context["bot_user_id"]),
        ).start()


if __name__ == "__main__":
    print("Cordy relay online.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
