# PARSeg-TAM-NT: the no-text control for TAM's 48.73. From scratch 160k.
#
# Identical to parsegtam in every respect except tam_use_text=False: the
# text input to the metric is zeroed, so w = 1 + 0.5*tanh(r) -- a freely
# learned per-class diagonal metric with NO language structure.
#
# Decides the thesis's text claim:
#   TAM >> TAM-NT  -> language content is load-bearing (the win is a TEXT
#                     win; W(desc_mean) coupling across description-similar
#                     classes is doing real work);
#   TAM ~= TAM-NT  -> any per-class diagonal metric helps; the honest
#                     claim becomes "per-class metric adaptation", text
#                     demoted to initialization flavor.
# Either outcome is publishable; running it before writing is mandatory.
_base_ = ['./parsegtam_ade20k_160k-512x512.py']

model = dict(
    decode_head=dict(
        args=dict(
            tam_use_text=False,
        )))
