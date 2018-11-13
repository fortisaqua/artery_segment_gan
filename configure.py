import json

'''
decay_step = 16 / (config['train_amount'] / 2)
'''

class Condifure:
    def __init__(self, confPath = ""):
        self.meta = {}
        self.confPath = confPath
        if len(confPath) == 0:
            self.meta["batchSize"] = 2
            self.meta["learningRateOrigin"] = 0.0001
            self.meta["decayRate"] = 0.9
            self.meta["blockShape"] = [256,256,8]
            self.meta["epochWalked"] = 0
            self.meta["stepWalked"] = 0
            self.meta["predictThreshold"] = 0.6
            self.meta["maxEpoch"] = 2000
            self.meta["updateEpoch"] = 5
            self.meta["testEpoch"] = 1
            self.meta["recordStep"] = 10
            self.meta["testStep"] = 20
            self.meta["saveStep"] = 50
            self.meta["clampThickness"] = 20
            self.meta["testDataPath"] = ["/opt/liver_data_set/3Dircadb1.17/original"]
            self.meta["metaDataPath"] = "/opt/artery_extraction/data_meta_liver.pkl"
            self.meta["modelPath"] = "/opt/deepLearningOutput/liver/models/"
            self.meta["resultPath"] = "/opt/deepLearningOutput/liver/periodical_result/"
            self.meta["sumPathTrain"] = "/opt/deepLearningOutput/liver/train_sum/"
            self.meta["sumPathTest"] = "/opt/deepLearningOutput/liver/test_sum/"
            self.meta["learningRate"] = self.meta["learningRateOrigin"]
            self.meta["outputEpoch"] = self.meta["testEpoch"] * 20
            self.meta["network"] = {}
            self.meta["network"]["generatorOriginSize"] = 24
            self.meta["network"]["denseBlockGrowth"] = 24
            self.meta["network"]["denseBlockDepth"] = 4
            self.meta["data"] = {}
            self.meta["data"]["sampleAmount"] = 8
            self.meta["data"]["testAmount"] = 2
        else:
            self.loadConf()

    def outputConf(self, confPath = ""):
        with open(confPath, "w") as f:
            json.dump(self.meta, f)

    def loadConf(self):
        with open(self.confPath, "r") as f:
            self.meta = json.load(f)

# conf = Condifure()
# conf.outputConf("./conf.json")
# print("")