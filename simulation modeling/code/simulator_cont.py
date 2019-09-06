import numpy as np
import pandas as pd

class Simulator:

    def __init__(self, n, x0, min_effort, max_effort, a, alpha, num_groups):
        self.num_groups = num_groups
        self.n = n
        self.x0 = x0
        self.a = a
        self.alpha = alpha

        # BUILD ARRAYS
        self.pop_size = num_groups * n
        # Group IDs
        self.group_ids = np.repeat(np.array(range(0, self.num_groups), dtype=np.int32), self.n)
        # Row IDs
        self.student_ids = np.array(range(self.pop_size), dtype=np.int32)
        # Strats
        hardworker_num = round(self.pop_size * self.x0)

        self.current_strats = min_effort + (np.random.beta(a=5, b=5, size=self.pop_size) * max_effort)

        self.reset_temp_columns()
        self.shuffle_students()

    def advance(self):
        self.reset_temp_columns()
        self.shuffle_students()
        self.update_pairing_ids()
        self.update_payoffs()
        self.update_strats()
        return self.current_strats

    def reset_temp_columns(self):
        self.paring_ids = np.zeros(shape=self.pop_size, dtype=np.int32)
        self.old_strats = np.empty(shape=self.pop_size, dtype=np.float64) * np.nan
        self.marks = np.empty(shape=self.pop_size, dtype=np.float64) * np.nan
        self.efforts = np.empty(shape=self.pop_size, dtype=np.float64) * np.nan
        self.payoffs = np.empty(shape=self.pop_size, dtype=np.float64) * np.nan
        self.score_diffs = np.empty(shape=self.pop_size, dtype=np.float64) * np.nan

    def shuffle_students(self):
        permutation = np.random.permutation(self.pop_size)
        self.student_ids = self.student_ids[permutation]
        self.current_strats = self.current_strats[permutation]
        self.old_strats = self.old_strats[permutation]

    def update_payoffs(self):

        reshaped_efforts = np.reshape(a=self.current_strats, newshape=(self.num_groups, self.n))
        group_marks = np.sum(reshaped_efforts, axis=1) / self.n

        self.marks = group_marks[self.group_ids]
        self.payoffs = group_marks[self.group_ids] - (self.a * self.current_strats)

    def update_pairing_ids(self):
        unpaired = self.student_ids
        while unpaired.shape[0] > 0:
            self.paring_ids[unpaired] = self.student_ids[np.random.choice(self.student_ids, unpaired.shape[0])]
            unpaired = unpaired[np.where(np.equal(self.paring_ids[unpaired], unpaired))]

    def update_strats(self):
        self.score_diffs = self.payoffs[self.paring_ids] - self.payoffs
        update_indexes = np.where(
            np.logical_and(
                np.greater(self.alpha * (self.score_diffs), np.random.random(self.pop_size)),
                np.logical_not(np.isclose(self.payoffs, self.payoffs[self.paring_ids]))
            )
        )
        self.old_strats = np.copy(self.current_strats)
        self.current_strats[update_indexes] = self.current_strats[self.paring_ids][update_indexes]

    def __str__(self):
        columns = ('Group', 'StudentID', 'Strategy', 'PartnerID', 'Mark', 'Cost', 'Payoff', 'PayoffDiff', "PrevStrat")
        df = pd.DataFrame(
            data=np.vstack(
                (
                    self.group_ids,
                    self.student_ids,
                    self.current_strats,
                    self.paring_ids,
                    np.round(self.marks, 2),
                    np.round(self.efforts * self.a, 2),
                    np.round(self.payoffs, 2),
                    np.round(self.score_diffs, 2),
                    self.old_strats
                )
            ).transpose(),
            columns=columns,

        )
        df.Strategy = np.where(df.Strategy==0, 'Lazy', 'Hard')
        df.PrevStrat = np.where(df.PrevStrat==0, 'Lazy', 'Hard')
        df[['Group', 'StudentID', 'PartnerID']] = df[['Group','StudentID','PartnerID']].astype(dtype=int)
        return df.to_string(index=False, line_width=200, formatters={'Group':str,'StudentID':str,'PartnerID':str})

    def __repr__(self):
        return self.__str__()
