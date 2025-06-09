# SarahWalletVisualizer.py
# Graphical display of wallet data using matplotlib

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

WALLET_PATH = os.path.join("data", "crypto", "wallet.srh")


def load_wallet():
    if os.path.exists(WALLET_PATH):
        with open(WALLET_PATH, 'r') as f:
            return json.load(f)
    return None


def plot_balance(wallet):
    timestamps = []
    balances = []
    balance = 0.0

    for tx in wallet.get("transactions", []):
        if tx['type'] == 'receive':
            balance += tx['amount']
        elif tx['type'] == 'send':
            balance -= tx['amount']
        timestamps.append(datetime.fromisoformat(tx['timestamp']))
        balances.append(balance)

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, balances, marker='o', linestyle='-', color='gold')
    plt.title("SarahCoin Wallet Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("SRH Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    wallet = load_wallet()
    if wallet:
        plot_balance(wallet)
    else:
        print("[WalletVisualizer] No wallet file found.")
