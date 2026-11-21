"""합성 입력만으로 잔차 TCN 연산/메모리 준비 확인. 공모전 자료 접근 없음."""
import json
import time


def main():
    import torch
    from .sequence_residual import make_model, SPEC

    torch.manual_seed(2026)
    torch.set_num_threads(1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = make_model().to(device)
    x = torch.randn(64, SPEC["channels"], 24, device=device)
    y = torch.randn(64, 1, 24, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    assert torch.count_nonzero(model(x)).item() == 0
    start = time.perf_counter()
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = (.2*model(x)-y).abs().mean()
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    result = dict(torch_version=torch.__version__, device=str(device), synthetic_only=True,
                  parameters=sum(p.numel() for p in model.parameters()), steps=25,
                  seconds=round(time.perf_counter()-start, 3), competition_data_read=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
