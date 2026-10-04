"""Fresh unperturbed severe states, evaluated by the same physical replay."""
from evaluation.recovery_policy_eval import pressure

def evaluate_natural(env,net,mean,std,metadata,cohort,path):
    return pressure(env,net,mean,std,metadata,cohort,path)
