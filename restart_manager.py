from typing import List


class RestartManager:

    def __init__(self, idleness_limit, K):
        self.idleness_limit, self.K = idleness_limit, K
        self.utilization = {code: 0 for code in range(K)}

    def add_code(self, code: int):
        if code not in self.utilization:
            self.utilization[code] = 0
        self.K = len(self.utilization)

    def reset_count(self, selections: List[int]):
        for code in selections:
            self.utilization[code] = 0

    def find_restarts(self, selections: List[int]):
        self.reset_count(selections)
        refreshed = set(selections)
        for code in range(self.K):
            if code not in refreshed:
                self.utilization[code] = self.utilization[code] + 1
        ranked = sorted(
            self.utilization.items(), key=(lambda x: x[1]), reverse=True)
        return [
            code for code, since_used in ranked
            if since_used > self.idleness_limit][:len(selections)]
