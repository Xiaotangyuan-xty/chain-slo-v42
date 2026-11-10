"""Allow driver bookkeeping memory only, with no compute process or GPU work."""
def idle(gpu_line, processes, utilization):
    used = int(gpu_line.rsplit(',', 1)[-1].strip().split()[0])
    return not processes.strip() and used <= 16 and int(utilization) == 0
