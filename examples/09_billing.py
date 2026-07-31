"""Balance and itemized usage.

Both are scoped to the *account*, not to the key you authenticate with: the
history covers every key the account owns, including deleted ones.
"""

from nexara import Nexara

client = Nexara(api_key="mock-key")

balance = client.billing.balance()
print(f"balance: {balance.balance} {balance.currency} ({balance.rate_per_min} per min)")

# A rough runway, on plain transcription only — profanity_filter, roles and
# prompt each add a surcharge that rate_per_min does not include.
print(f"~{int(balance.balance / balance.rate_per_min)} minutes left at that rate")

# One page, newest first. Pagination is keyset, not offset: pass the previous
# page's next_cursor to walk backwards in time.
page = client.billing.usage(limit=3)
for item in page.items:
    # cost is None for rows written before per-request costs were recorded —
    # "unknown", not "free".
    cost = "  n/a" if item.cost is None else f"{item.cost:.2f} {page.currency}"
    print(f"{item.timestamp:%Y-%m-%d %H:%M}  {item.task:<10} {cost}  via {item.api_key.name}")
print(f"has_more={page.has_more} next_cursor={page.next_cursor}")

# iter_usage() does the paging for you. History is unbounded, so bound it.
total = sum(item.cost or 0 for item in client.billing.iter_usage(max_items=100))
print(f"last 100 calls cost {total:.2f} {page.currency}")
