class Track():
    def __init__(self, id, status, measurment,mean ,frame , covariance,descriptor):
        self.id = id
        self.status = status
        self.measurement = measurment
        self.counter = 0
        self.mean = mean
        self.frame = frame
        self.covariance = covariance
        self.descriptor = descriptor

    def reset(self):
        self.counter = 0

    def inc_count(self):
        self.counter+=1