import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video import transcribe_video

t = transcribe_video("data/sample_video.mp4")
print("Transcript:", t.text[:500])
print("Seconds:", t.duration_s)
print("Cost: $%.6f" % t.cost_usd)