import json
d = json.load(open("d:/Ticket-support-system/test_results.json"))
with open("d:/Ticket-support-system/results_short.txt", "w") as f:
    for k, v in d.items():
        status = "OK" if v == "OK" else "FAIL"
        f.write(f"{status}: {k}\n")
print("done")
