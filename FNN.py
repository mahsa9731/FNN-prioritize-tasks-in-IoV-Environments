import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#R 0.111111 0.999999
# =====================================================================

# 1. Fuzzy Membership Functions (Gaussian MFs to avoid zero gradient)

# =====================================================================



mfs = {

    'initial_task_priority': {'Low': [1.0, 0.5],  'Medium': [2.0, 0.5],  'High': [3.0, 0.5]},

    'time_slack': {
    'Low': [0.0, 1.5],
    'Medium': [7.5, 3.0],
    'High': [15.0, 2.5]
},

   'deadline': {
    'Low': [0.2, 1.5],
    'Medium': [7.5, 3.0],
    'High': [15.0, 2.5]
},
    'reliability': {
    'Low': [0.90, 0.025],
    'Medium': [0.95, 0.020],
    'High': [0.999999, 0.015]
},

    'ram_demand': {
    'Low': [0.001, 0.08],
    'Medium': [0.20, 0.10],
    'High': [0.49, 0.08]
},

    'speed': {
    'Low': [20.0, 15.0],
    'Medium': [70.0, 20.0],
    'High': [120.0, 15.0]
}

}


FEATURE_BOUNDS = torch.tensor([
    [1.0, 3.0],       # initial_task_priority
    [0.0, 15.0],      # time slack (seconds)
    [0.200, 15.0],   # deadline (seconds)
    [0.90, 0.999999],     # reliability
    [0.001, 0.49],   # ram_demand (GB)
    [20.0, 120.0]     # speed (km/h)
], dtype=torch.float32)

# =====================================================================

# 2. Gaussian Fuzzy Layer (Smooth Gradients everywhere)

# =====================================================================



class GaussianFuzzyLayer(nn.Module):



    def __init__(self, mfs_dict):

        super(GaussianFuzzyLayer, self).__init__()



        self.param_names = list(mfs_dict.keys())

        centers, sigmas = [], []



        for param in self.param_names:

            p_centers, p_sigmas = [], []



            for level in ['Low', 'Medium', 'High']:

                c, s = mfs_dict[param][level]

                p_centers.append(c)

                p_sigmas.append(s)



            centers.append(p_centers)

            sigmas.append(p_sigmas)



        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))

        self.raw_sigmas = nn.Parameter(torch.tensor(sigmas, dtype=torch.float32))



    def forward(self, x):



        safe_sigmas = torch.abs(self.raw_sigmas) + 1e-3

        x_exp = x.unsqueeze(-1)



        # Gaussian MF formula: exp(-0.5 * ((x - c) / sigma)^2)

        membership = torch.exp(

            -0.5 * torch.pow((x_exp - self.centers) / safe_sigmas, 2)

        )



        return membership  # Output shape: (batch_size, 6, 3)





# =====================================================================

# 3. Robust ANFIS / Neuro-Fuzzy Architecture

# =====================================================================



class AutonomousTaskANFIS(nn.Module):



    def __init__(self, mfs_dict, bounds):

        super(AutonomousTaskANFIS, self).__init__()



        self.fuzzy_layer = GaussianFuzzyLayer(mfs_dict)

        self.num_params = len(mfs_dict)



        self.register_buffer('min_bounds', bounds[:, 0].unsqueeze(0))

        self.register_buffer('max_bounds', bounds[:, 1].unsqueeze(0))



        # FC Consequent Layer with Sigmoid activation scaled to [0, 10]

        self.fc1 = nn.Linear(self.num_params * 3, 32)

        self.fc2 = nn.Linear(32, 16)

        self.out_layer = nn.Linear(16, 1)



        self.activation = nn.SiLU()  # Smooth non-zero gradient activation



    def forward(self, x):



        # 1. Feature Fuzzification using raw scale

        memberships = self.fuzzy_layer(x)



        # 2. Normalization of inputs for dense layer stability

        x_norm = (x - self.min_bounds) / (

            self.max_bounds - self.min_bounds + 1e-8

        )



        x_norm_exp = x_norm.unsqueeze(-1)



        # 3. Conjunction / Fuzzy Fusing

        fused_features = (x_norm_exp * memberships).view(x.size(0), -1)



        # 4. Neural Defuzzification

        h1 = self.activation(self.fc1(fused_features))

        h2 = self.activation(self.fc2(h1))



        out = torch.sigmoid(self.out_layer(h2)) * 10.0



        return out





# =====================================================================

# 4. Expert Ground Truth Engine

# =====================================================================



class ExpertRuleBase:



    def __init__(self):



        self.weights = {

            'initial_task_priority': 0.35,

            'time_slack': 0.21,

            'deadline': 0.175,

            'reliability': 0.14,

            'ram_demand': 0.105,

            'speed': 0.02

        }



    def evaluate_expert_priority(self, X_tensor):



        initial_task_priority = X_tensor[:, 0]

        time_slack  = X_tensor[:, 1]

        deadline    = X_tensor[:, 2]

        reliability = X_tensor[:, 3]

        ram         = X_tensor[:, 4]

        speed       = X_tensor[:, 5]



        score_slack = (15.000 - time_slack) / 15.000

        score_deadline = (15.000 - deadline) / 14.800

        score_reliability = (reliability - 0.90) / 0.099999

        score_ram = (ram - 0.001) / (0.49 - 0.001)

        score_initial_priority = (initial_task_priority - 1.0) / 2.0

        score_speed = (speed - 20.0) / 100.0



        weighted_score = (

            score_slack       * self.weights['time_slack'] +

            score_deadline    * self.weights['deadline'] +

            score_reliability * self.weights['reliability'] +

            score_ram         * self.weights['ram_demand'] +

            score_initial_priority * self.weights['initial_task_priority'] +

            score_speed       * self.weights['speed']

        )



        return (weighted_score * 10.0).unsqueeze(1)





# =====================================================================

# 5. Dataset Generation & Training Process

# =====================================================================



def generate_expert_dataset(num_samples=6000):



    torch.manual_seed(42)


    exec_time = torch.FloatTensor(
        num_samples, 1
        ).uniform_(0.010, 0.200)  
    
    deadline = torch.FloatTensor(num_samples, 1).uniform_(0.200, 15.000)

    reliability = torch.FloatTensor(
    num_samples, 1
    ).uniform_(0.90, 0.999999)


    size_mb = torch.FloatTensor(
    num_samples, 1
          ).uniform_(1.0, 500.0)

    metadata = 0.05

    ram = (size_mb + metadata) / 1024.0

    initial_task_priority = torch.randint(1, 4, (num_samples, 1)).float()



    speed = torch.FloatTensor(
    num_samples, 1
    ).uniform_(20.0, 120.0)


    time_slack = torch.clamp(deadline - exec_time, min=0.0) #DEADLINE SHOULD BE HIGHER!


    X = torch.cat([

        initial_task_priority,

        time_slack,

        deadline,

        reliability,

        ram,

        speed

    ], dim=1)



    expert_engine = ExpertRuleBase()

    y = expert_engine.evaluate_expert_priority(X)



    return X, y





def train_anfis(model, epochs=500, lr=0.005):


    loss_history = []
    X_train, y_train = generate_expert_dataset(num_samples=6000)



    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)



    print("==================================================")

    print("Starting Fixed ANFIS Model Training Process")

    print("==================================================")



    model.train()



    for epoch in range(epochs):



        optimizer.zero_grad()



        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        loss_history.append(loss.item())

        loss.backward()

        optimizer.step()



        if (epoch + 1) % 50 == 0 or epoch == 0:

            print(

                f"Epoch [{epoch+1:3d}/{epochs}] ---> "

                f"Loss (MSE): {loss.item():.6f}"

            )



    print("==================================================\n")

    # ============================================================
    # Scientific Training Loss Curve
    # ============================================================

    plt.figure(figsize=(10, 6))

    epochs_range = range(1, epochs + 1)

    # Main loss curve
    plt.plot(
        epochs_range,
        loss_history,
        linewidth=2.0,
        label="Training Loss"
    )

    # Moving average for a smoother trend
    window = 15

    if len(loss_history) >= window:
        moving_avg = np.convolve(
            loss_history,
            np.ones(window) / window,
            mode='valid'
        )

        plt.plot(
            range(window, epochs + 1),
            moving_avg,
            linewidth=2.5,
            linestyle='--',
            label=f"Moving Average ({window} epochs)"
        )

    plt.xlabel(
        "Epoch",
        fontsize=13,
        fontweight='bold'
    )

    plt.ylabel(
        "Mean Squared Error (MSE)",
        fontsize=13,
        fontweight='bold'
    )

    plt.title(
        "ANFIS Training Loss",
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    plt.grid(
        True,
        linestyle='--',
        linewidth=0.7,
        alpha=0.35
    )

    plt.legend(
        fontsize=11,
        frameon=True
    )

    plt.tick_params(
        axis='both',
        labelsize=11
    )

    plt.tight_layout()

    LOSS_PLOT_PATH = "anfis_training_loss.png"

    plt.savefig(
        LOSS_PLOT_PATH,
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Scientific loss curve saved to: {LOSS_PLOT_PATH}")
    
    return model


# =====================================================================

# 6. Fixed Model Loading / Training

# =====================================================================

MODEL_PATH = "anfis_task_priority_model.pth"


def save_anfis_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"ANFIS model saved to: {path}")


def load_anfis_model(model, path=MODEL_PATH):
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
    print(f"ANFIS model loaded from: {path}")
    return model



anfis_model = AutonomousTaskANFIS(
    mfs,
    FEATURE_BOUNDS
)

if os.path.exists(MODEL_PATH):

    # Load previously trained model
    anfis_model = load_anfis_model(
        anfis_model,
        MODEL_PATH
    )

else:

    # Train model only once
    anfis_model = train_anfis(
        anfis_model,
        epochs=500,
        lr=0.005
    )

    # Save trained model
    save_anfis_model(
        anfis_model,
        MODEL_PATH
    )

anfis_model.eval()

# =====================================================================
# 7. Interface for DAG Task Generator
# =====================================================================

def get_task_priority(
    current_deadline=None,
    current_slack=None,
    reliability=None,
    ram=None,
    initial_task_priority=None,
    speed=None,
    deadline=None,
    exec_time=None
):

    # Compatibility with DAG_Generator
    if current_deadline is None:
        current_deadline = deadline

    if current_slack is None:
        current_slack = max(
            0.0,
            current_deadline - exec_time
        )

    input_data = torch.tensor([[
        initial_task_priority,
        current_slack,
        current_deadline,
        reliability,
        ram,
        speed
    ]], dtype=torch.float32)

    anfis_model.eval()

    with torch.no_grad():
        priority = anfis_model(input_data).item()

    return priority


# if __name__ == "__main__":

#     def make_input(deadline, reliability, exec_time, ram, initial_task_priority, speed):



#         slack = max(0.0, deadline - exec_time)



#         return torch.tensor(

#             [[initial_task_priority, slack, deadline, reliability, ram, speed]],

#             dtype=torch.float32

#         )



#     # Emergency vehicle: Initial task priority 3

#     critical_task = make_input(

#         deadline=0.10,

#         reliability=0.98,

#         exec_time=0.02,

#         ram=16.0,

#         initial_task_priority=3,

#         speed=130.0

#     )



#     # Personal vehicle: Initial task priority 1

#     normal_task = make_input(

#         deadline=4.80,

#         reliability=0.20,

#         exec_time=0.45,

#         ram=0.01,

#         initial_task_priority=1,

#         speed=15.0

#     )



#     # Public transportation / taxi: Initial task priority 2

#     medium_task = make_input(

#         deadline=2.50,

#         reliability=0.70,

#         exec_time=0.20,

#         ram=8.0,

#         initial_task_priority=2,

#         speed=70.0

#     )



#     random_task = make_input(

#         deadline=0.2,

#         reliability=0.70,

#         exec_time=0.1,

#         ram=4,

#         initial_task_priority=1,

#         speed=50.0

#     )







#     anfis_model.eval()



#     with torch.no_grad():



#         p_critical = anfis_model(critical_task).item()

#         p_normal = anfis_model(normal_task).item()

#         p_medium = anfis_model(medium_task).item()

#         p_random = anfis_model(random_task).item()



#     print(f"Critical Task Priority Score (ANFIS Prediction): {p_critical:.2f} / 10.0")

#     print(f"Normal Task Priority Score (ANFIS Prediction):   {p_normal:.2f} / 10.0")

#     print(f"Medium Task Priority Score (ANFIS Prediction):   {p_medium:.2f} / 10.0")

#     print(f"Random Task Priority Score (ANFIS Prediction): {p_random:.2f} / 10.0")



#Critical Task Priority Score (ANFIS Prediction): 8.80 / 10.0

#Normal Task Priority Score (ANFIS Prediction):   1.09 / 10.0

#Medium Task Priority Score (ANFIS Prediction):   5.19 / 10.0

# Task Priority Score (ANFIS Prediction): 6.97 / 10.0