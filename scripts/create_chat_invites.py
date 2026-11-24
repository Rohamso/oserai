"""Generate a private sign-in code for each named OserAI pilot account."""

import argparse
import hashlib
import json
import secrets

from chat_service.pilot import ACCOUNT_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accounts", nargs="+", metavar="ACCOUNT")
    args = parser.parse_args()
    if any(not ACCOUNT_ID.fullmatch(account) for account in args.accounts):
        parser.error("Use 1–40 lowercase letters, numbers, underscores or hyphens per account. Start with a letter or number.")
    if len(set(args.accounts)) != len(args.accounts):
        parser.error("Account IDs must be unique.")

    digests = {}
    print("Private access codes. Share each code only with its tester.")
    for account in args.accounts:
        code = secrets.token_urlsafe(32)
        digests[account] = hashlib.sha256(code.encode()).hexdigest()
        print(f"{account}  {code}")
    print("\nValue for OSERAI_CHAT_PILOT_ACCOUNTS in Render")
    print(json.dumps(digests, separators=(",", ":")))


if __name__ == "__main__":
    main()
