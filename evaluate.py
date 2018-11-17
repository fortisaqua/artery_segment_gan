import numpy as np

class Indicator:
    def __init__(self):
        pass
    def calculate(self, RS, TS):
        pass

class SA(Indicator):
    def __init__(self):
        Indicator.__init__(self)

    def calculate(self, RS, TS):
        missing = np.float32((RS - TS) > 0)
        ans = 1 - np.sum(missing) / np.sum(RS)
        return ans

class OR(Indicator):
    def __init__(self):
        Indicator.__init__(self)

    def calculate(self, RS, TS):
        overSeg = np.float32((TS - RS) > 0)
        ans = np.sum(overSeg) / (np.sum(overSeg) + np.sum(RS))
        return ans

class UR(Indicator):
    def __init__(self):
        Indicator.__init__(self)

    def calculate(self, RS, TS):
        missing = np.float32((RS - TS) > 0)
        overSeg = np.float32((TS - RS) > 0)
        ans = np.sum(missing) / (np.sum(RS) + np.sum(overSeg))
        return ans

class IOU(Indicator):
    def __init__(self):
        Indicator.__init__(self)

    def calculate(self, RS, TS):
        ans = 2 * np.sum(np.abs(RS * TS)) / np.sum(np.abs(RS) + np.abs(TS))
        return ans

class Evaluator:
    def __init__(self):
        self.evaluators = {}
        self.evaluators["SA"] = SA()
        self.evaluators["OS"] = OR()
        self.evaluators["UR"] = UR()
        self.evaluators["IOU"] = IOU()

    def Evaluate(self, RS, TS, keys):
        '''
        :param RS: ground truth
        :param TS: Prediction
        :param keys: indicators' names
        :return: a dictionary with evaluate results
        '''
        results = {}
        for eName in keys:
            results[eName] = self.evaluators[eName].calculate(RS, TS)
        return results