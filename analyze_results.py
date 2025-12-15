import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

experiments = ['vanilla', 'rope', 'swiglu', 'rmsnorm']
results = {}

for exp in experiments:
    path = os.path.join('experiments', exp)
    event_files = glob.glob(os.path.join(path, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event file found for {exp}")
        continue
    
    # Use the latest event file
    event_file = max(event_files, key=os.path.getmtime)
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    try:
        # We want the training loss at step 100
        # "loss/step" is logged every log_interval (10)
        # "loss/train" is logged every eval_interval (100)
        
        # Let's try to get the exact value at step 100 using 'loss/train' which is cleaner (eval loss)
        # or 'val/loss' 
        # But user asked for "tb losses", which usually implies the training curve.
        # Let's fetch 'loss/train' (calculated at eval) and 'loss/val' at step 100.
        
        train_losses = ea.Scalars('loss/train')
        val_losses = ea.Scalars('loss/val')
        
        # Find step 100
        t_loss = next((x.value for x in train_losses if x.step == 100), None)
        v_loss = next((x.value for x in val_losses if x.step == 100), None)
        
        results[exp] = {'train': t_loss, 'val': v_loss}
        
    except KeyError:
        print(f"Could not find loss scalar in {exp}")

print(f"{'Experiment':<15} {'Train Loss':<15} {'Val Loss':<15}")
print("-" * 45)
for exp in experiments:
    res = results.get(exp, {})
    t = f"{res.get('train', 'N/A'):.4f}" if res.get('train') else "N/A"
    v = f"{res.get('val', 'N/A'):.4f}" if res.get('val') else "N/A"
    print(f"{exp:<15} {t:<15} {v:<15}")
