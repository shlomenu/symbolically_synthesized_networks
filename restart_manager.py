from typing import List, Dict, Tuple, Optional


class RestartManager:

    def __init__(self, idleness_limit, K, contextual=False):
        self.idleness_limit, self.K, self.contextual = idleness_limit, K, contextual
        if self.contextual:
            self.contextual_utilization: Dict[int, Tuple[int, int, Tuple[Optional[int], Optional[int]], int]] = {
                code: (0, 0, (None, None), 0) for code in range(K)}
        else:
            self.utilization = {code: 0 for code in range(K)}

    def add_code(self, code):
        if hasattr(self, "contextual_utilization") and code not in self.contextual_utilization:
            self.contextual_utilization[code] = (0, 0, (None, None), 0)
        elif hasattr(self, "utilization") and code not in self.utilization:
            self.utilization[code] = 0

    def find_restarts(self, commands: List[int]):
        if self.contextual:
            for i, code in enumerate(commands):
                (_, since_novel_usage, neighbors,
                 position) = self.contextual_utilization[code]
                new_neighbors = (None if i == 0 else commands[i - 1],
                                 None if i == len(commands) - 1 else commands[i +
                                                                              1])
                new_position = i
                if neighbors == new_neighbors and position == new_position:
                    new_since_novel_usage = since_novel_usage + 1
                else:
                    new_since_novel_usage = 0
                self.contextual_utilization[code] = (0,
                                                     new_since_novel_usage,
                                                     new_neighbors,
                                                     new_position)
            refreshed = set(commands)
            for code in range(self.K):
                if code not in refreshed:
                    (since_used, since_novel_usage, neighbors,
                     position) = self.contextual_utilization[code]
                    self.contextual_utilization[code] = (since_used + 1, since_novel_usage + 1,
                                                         neighbors, position)
            return [
                code
                for code, (since_used, since_novel_usage, _,
                           _) in self.contextual_utilization.items()
                if since_used > self.idleness_limit
                or since_novel_usage > self.idleness_limit
            ]
        else:
            for code in commands:
                self.utilization[code] = 0
            refreshed = set(commands)
            for code in range(self.K):
                if code not in refreshed:
                    self.utilization[code] = self.utilization[code] + 1
            return [
                code for code, since_used in self.utilization.items()
                if since_used > self.idleness_limit
            ]
