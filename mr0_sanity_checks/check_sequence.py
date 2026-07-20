import MRzeroCore as mr0
import pypulseq as pp
import torch

from sys import argv

seq_file = argv[1]
viz_amount = float(argv[2])

TR = 2e-3

sequence = mr0.Sequence.import_file(seq_file)
pp_sequence = pp.Sequence()
pp_sequence.read(seq_file)

assert viz_amount < sequence.get_duration()

adc_events = torch.cat([r.adc_usage for r in sequence])
adc_event_dists = torch.argwhere(torch.diff(adc_events))
spoke_length = (adc_event_dists[1] - adc_event_dists[0]).item()
num_spokes = sequence.get_kspace().shape[0] // spoke_length
kspace_times = pp_sequence.calculate_kspace()[4]

print(f"Sequence length: {len(sequence)}")
print(f"Spoke length: {spoke_length}")
print(f"Number of spokes: {num_spokes}")

acq_start_time = kspace_times[adc_event_dists[0]]

mr0.util.pulseq_plot(
    seq=pp_sequence,
    time_range=(acq_start_time, acq_start_time + viz_amount),
    time_disp='s',
    show_blocks=True
)
