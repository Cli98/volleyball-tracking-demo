import numpy as np


class volleyball:
    def __init__(self, array):
        x, y, r, a = array
        self.coord = [[x, y]]
        self.property = [[r, a]]
        # There are three state: initialized (0), static (1), directed (2)
        self.status = 0
        self.R = 60
        self.frame_count = a
        self.next_x = -1
        self.next_y = -1

    def __len__(self):
        return len(self.coord)

    def compute_dist(self, idx1, idx2):
        dx = self.coord[idx1][0] - self.coord[idx2][0]
        dy = self.coord[idx1][1] - self.coord[idx2][1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def fit_within_group(self, idx1, idx2):
        distance = self.compute_dist(idx1, idx2)
        return distance < self.R, distance

    def fit(self, idx, x, y):
        dx = self.coord[idx][0] - x
        dy = self.coord[idx][1] - y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance<self.R, distance

    def add(self, array):
        # Once we call this function, we shoudl have at least one point.
        x, y, r, a = array
        self.coord.append([x, y])
        self.property.append([r, a])
        if len(self.coord) > 2:
            dx1, dy1 = self.coord[-1][0] - self.coord[-2][0], self.coord[-1][1] - self.coord[-2][1]
            dx2, dy2 = self.coord[-2][0] - self.coord[-3][0], self.coord[-2][1] - self.coord[-3][1]
            d1 = self.compute_dist(-1, -2)
            d2 = self.compute_dist(-2, -3)

            if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d1 > 5 and d2 > 5:
                self.status = 2
            else:
                self.status = 1
        else:
            self.status = 2

    def predict(self):
        # Project next point given history
        npos = np.array(self.coord)
        total_length = len(npos) + 1
        idx = np.array((1, total_length))

        kx = np.polyfit(idx, npos[:, 0], 1)
        project_x = np.poly1d(kx)
        ky = np.polyfit(idx, npos[:, 1], 1)
        project_y = np.poly1d(ky)

        self.next_x, self.next_y = project_x, project_y
        return project_x, project_y
