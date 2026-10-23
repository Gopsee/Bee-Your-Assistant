# Bee, your Assistant

A warm, patient companion for people living with memory loss and their
caregivers, built on the Bee wearable.

## Run

```bash
# 1. Pick a random token and start the Bee MCP server with it
export BEE_MCP_TOKEN=$(openssl rand -hex 32)
bee mcp serve-http --token "$BEE_MCP_TOKEN"

# 2. In another shell with the same BEE_MCP_TOKEN, start the app
python server.py                      # http://127.0.0.1:8000 (mobile view at /mobile)

# Reachable from a phone on the same network: a password is required
BEE_PASSWORD='choose-one' python server.py --host 0.0.0.0
```

The `bee` CLI must be logged in (`bee login`). AWS credentials with access to
Amazon Bedrock must be set up as usual for boto3.

## Privacy & compliance

**Who it is for.** Adults living with memory loss and the family members or
caregivers who support them. It is not intended for, or designed for use by,
children under 13.

**Not a medical device.** Bee gives companionship and reminders. It does not
diagnose, treat or monitor any condition, and it never confirms medication
doses it hasn't been told about.

**What data goes where**

| Data | Where it lives | Who can see it |
|---|---|---|
| Audio, transcripts, summaries, facts, to-dos captured by the wearable | The user's Bee account (Bee's cloud) | Bee, under Bee's own privacy policy |
| Local copy of every Bee conversation and transcript | `bee_conversation/` on this computer | Anyone with access to this computer |
| Long-term memory, session memory, things it is still learning, "likes to know" | `memory/` on this computer | Anyone with access to this computer |
| Chat logs | `logs/` on this computer | Anyone with access to this computer |
| Chat messages, plus the memory and Bee data needed to answer them | Sent to Amazon Bedrock (Amazon Nova) for each model call; not stored by this app beyond the files above | AWS, under the AWS service terms |
| Topic screening of each message | Runs locally (no network) | Nobody else |

- Local files are plain JSON and Markdown and are **not encrypted**. Use full
  disk encryption on the computer that runs the app.
- The app talks to the Bee account only through the `bee` CLI and the Bee MCP
  server, both on this computer. Tokens are read from environment variables,
  never stored in the code.
- The server listens on `127.0.0.1` only by default. Opened to the network
  (`--host 0.0.0.0`) it refuses to start without `BEE_PASSWORD`. That
  password is sent with HTTP Basic auth over plain HTTP, so only use it on a
  trusted network.
- **Delete my data** (Profile in the mobile view) deletes `memory/`, `logs/`
  and `bee_conversation/`. Data in the Bee account is deleted in the Bee app;
  anything still there is copied again on the next sync.

**Not HIPAA-covered.** This is a hackathon project, not a HIPAA-covered
deployment. Clinical use (for example by a care provider) would need at
least: a Business Associate Agreement (BAA) with AWS and Bedrock limited to
HIPAA-eligible services, a BAA with Bee, encryption of all stored data, user
logins with access control and audit logs, and HTTPS.

**Demo data is fictional.** Walt, Margaret, Daniel and everyone in
`demo_profiles/`, `demo_memory/` and `demo-conversation/` are made-up people
created to test the app. Any resemblance to real people is coincidental.
