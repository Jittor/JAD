import jittor

def IOU(intputs, targets):
    numerator = (intputs * targets).to(jittor.int32).sum(dim=1)
    denominator = intputs.to(jittor.int32).sum(dim=1) + targets.to(jittor.int32).sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss, numerator, denominator
