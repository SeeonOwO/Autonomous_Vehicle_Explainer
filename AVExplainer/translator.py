action_labels = ['move forward', 'stop/slow down', 'turn left', 'turn right']
reason_labels = [
    'Forward - follow traffic',
    'Forward - the road is clear',
    'Forward - the traffic light is green',
    'Stop/slow down - obstacle: car',
    'Stop/slow down - obstacle: person/pedestrian',
    'Stop/slow down - obstacle: rider',
    'Stop/slow down - obstacle: others',
    'Stop/slow down - the traffic light',
    'Stop/slow down - the traffic sign',
    'Turn left - front car turning left',
    'Turn left - on the left-turn lane',
    'Turn left - traffic light allows',
    'Turn right - front car turning right',
    'Turn right - on the right-turn lane',
    'Turn right - traffic light allows',
    "Can't turn left - obstacles on the left lane",
    "Can't turn left - no lane on the left",
    "Can't turn left - solid line on the left",
    "Can't turn right - obstacles on the right lane",
    "Can't turn right - no lane on the right",
    "Can't turn right - solid line on the left"
]

def generate_explanation(action_idx, reason_idx):
    action = action_labels[action_idx]
    reason = reason_labels[reason_idx]
    explanation = f"{action.capitalize()}! {reason}"
    return explanation

print(generate_explanation(3,13))