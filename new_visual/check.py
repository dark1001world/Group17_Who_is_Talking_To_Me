#python3 -c "
import sys
sys.path.insert(0, '/DATA/G17/Group17_Who_is_Talking_To_Me/new_visual')
from model import build_model

for n in [0, 1, 2, 3, 4]:
    model = build_model('VideoSwinV2TTM', variant='swin3d_s',
                        pretrained=False, num_classes=2, dropout=0.2,
                        freeze_stages=n)
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'freeze_stages={n}: trainable={t/1e6:.1f}M  frozen={f/1e6:.1f}M')
