import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

experiments = ['vanilla', 'all_features', 'vanilla_fineweb', 'all_features_fineweb', 'rope', 'swiglu', 'rmsnorm']
target_steps = [100, 200, 300, 1000, 2000]
results = {}

for exp in experiments:
    path = os.path.join('experiments', exp)
    event_files = glob.glob(os.path.join(path, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event file found for {exp}")
        continue
    
    event_file = max(event_files, key=os.path.getmtime)
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    results[exp] = {}
    try:
        train_losses = ea.Scalars('loss/train')
        val_losses = ea.Scalars('loss/val')
        
        for step in target_steps:
            t_loss = next((x.value for x in train_losses if x.step == step), None)
            v_loss = next((x.value for x in val_losses if x.step == step), None)
            results[exp][step] = {'train': t_loss, 'val': v_loss}
        
    except KeyError:
        print(f"Could not find loss scalar in {exp}")

print(f"{'Experiment':<15} {'Step':<10} {'Train Loss':<15} {'Val Loss':<15}")
print("-" * 60)
for exp in experiments:
    for step in target_steps:
        res = results.get(exp, {}).get(step, {})
        t = f"{res.get('train', 'N/A'):.4f}" if res.get('train') else "N/A"
        v = f"{res.get('val', 'N/A'):.4f}" if res.get('val') else "N/A"
        print(f"{exp:<15} {step:<10} {t:<15} {v:<15}")
