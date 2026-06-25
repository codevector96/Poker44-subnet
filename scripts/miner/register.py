#!/usr/bin/env python3
"""Register a miner hotkey on a subnet via the bittensor SDK.

Workaround for `btcli subnet register` failing with
``'bool' object has no attribute 'metadata'`` when the installed bittensor SDK
(10.x) is newer than the pinned bittensor-cli (9.20.0). The SDK itself works, so
we submit the recycle (burned) registration directly.

Run on the miner host inside the venv:
    python scripts/miner/register.py --wallet <coldkey> --hotkey <hotkey>

You will be prompted for the coldkey password to sign the registration.
"""
from __future__ import annotations

import argparse

import bittensor as bt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True, help="coldkey wallet name")
    ap.add_argument("--hotkey", required=True, help="hotkey name")
    ap.add_argument("--netuid", type=int, default=126)
    ap.add_argument("--network", default="finney")
    args = ap.parse_args()

    st = bt.Subtensor(network=args.network)
    w = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    hk = w.hotkey.ss58_address
    ck = w.coldkeypub.ss58_address
    print(f"coldkey: {ck}")
    print(f"hotkey:  {hk}")

    if st.is_hotkey_registered(netuid=args.netuid, hotkey_ss58=hk):
        print(f"Already registered on netuid {args.netuid}. Nothing to do.")
        return 0

    bal = st.get_balance(ck)
    cost = st.recycle(args.netuid)
    print(f"balance: {bal}  |  recycle cost: {cost}")
    if bal < cost:
        print("ERROR: insufficient balance to cover the recycle cost.")
        return 1

    print(f"Submitting burned registration on netuid {args.netuid} "
          f"(you'll be asked for the coldkey password)...")
    resp = st.burned_register(
        wallet=w,
        netuid=args.netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    print("extrinsic response:", resp)

    if st.is_hotkey_registered(netuid=args.netuid, hotkey_ss58=hk):
        try:
            uid = st.metagraph(args.netuid).hotkeys.index(hk)
            print(f"SUCCESS — registered. Your miner UID on netuid {args.netuid} is {uid}.")
        except (ValueError, Exception):
            print("SUCCESS — registered. UID not visible yet (metagraph sync lag).")
        return 0

    print("Registration did not take effect; check the response above and retry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
