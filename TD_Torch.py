# With Keras
inp=Input((None,13))
out=TimeDistributed(BatchNormalization())(inp)
out=LSTM(512,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)(out)
out=LSTM(512,dropout=0.1,recurrent_dropout=0.1,return_sequences=False)(out)
out=BatchNormalization()(out)
out=Dense(units=1024)(out)
out=Activation('relu')(out)
out=Dropout(0.1)(out)
out=Dense(units=512)(out)
out=Activation('relu')(out)
out=Dropout(0.1)(out)
out=Dense(units=256)(out)
out=Activation('relu')(out)
out=Dropout(0.1)(out)
out=Dense(units=128)(out)
out=Activation('relu')(out)
out=Dropout(0.1)(out)
out=Dense(units=39)(out)
out=Activation('softmax')(out)
Final_net=Model(inputs=inp,outputs=out)

# With PyTorch
import torch.nn as nn

class TD_LSTM(nn.Module):
    def init(self):
    super(TD_LSTM, self).init()
    # 1D CovNet for learning the Spectral features
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=(32,))
    self.bn1 = nn.BatchNorm1d(128)
    self.maxpool1 = nn.MaxPool1d(kernel_size=1, stride=97)
    self.dropout1 = nn.Dropout(0.3)
    # 1D LSTM for learning the temporal aggregation
    self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, dropout=0.3)
    # Fully Connected layer
    #self.fc3 = nn.Linear(128, 128)
    #self.bn3 = nn.BatchNorm1d(128)
    # Get posterior probability for target event class
    self.fc4 = nn.Linear(128, 1)
    self.timedist = TimeDistributed(self.fc4)



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y