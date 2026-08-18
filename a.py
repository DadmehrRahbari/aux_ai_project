from federated.client import FLClient
from aux_ai.monitor import AuxAIMonitor
c = FLClient(AuxAIMonitor(input_dim=526))
print(hasattr(c, 'local_train'))  # must print True
print(hasattr(c, 'criterion'))    # must print True