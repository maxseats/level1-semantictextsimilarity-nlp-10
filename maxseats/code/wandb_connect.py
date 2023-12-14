import subprocess
import wandb



######################################################################
#디폴트 : klue/roberta-small, 16, 1, True, 1e-5
#모델이름, 에폭 불러오기 -> sweep 이름 설정용
from train import one_model_name
from train import three_max_epoch
######################################################################



# WandB sweep configuration을 정의합니다.
sweep_config = {
        "name": f"{one_model_name}, epoch:{three_max_epoch}",  # Sweep의 이름
        "method": "grid",    # 탐색 방법 (grid 또는 random)
        "metric": {"goal": "maximize", "name": "val_pearson"},  # 최적화할 메트릭 설정
        "parameters": {
            "learning_rate": {"values": [0.001, 0.01, 0.1]},
            "batch_size": {"values": [8, 16, 32]},
        },
    }

# WandB sweep를 초기화하고 sweep ID를 가져옵니다.
sweep_id = wandb.sweep(sweep=sweep_config, project="maxseats")

# WandB sweep를 실행합니다.
wandb.agent(sweep_id, function=lambda: subprocess.call(['python', 'train.py']))


# 최적의 하이퍼파라미터 조합으로 스크립트를 실행합니다.
best_run = wandb.runs(path='all', filters={'sweep': sweep_id, 'state': 'finished', 'config.val_pearson': 'max'}).pop()
best_config = best_run.config
subprocess.call(['python', 'train.py', f'--learning_rate={best_config.learning_rate}', f'--batch_size={best_config.batch_size}'])
