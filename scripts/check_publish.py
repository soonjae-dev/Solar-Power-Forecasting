"""공개 전 점검: 올라갈 파일에 데이터·인증 정보·좌표·발전소 식별자가 섞였는지 검사한다.

    python scripts/check_publish.py          # git 저장소면 추적 대상, 아니면 .gitignore 허용 목록 기준
    python scripts/check_publish.py --list   # 검사 대상 파일 목록만 출력

발견 내용은 '파일:줄 번호 + 검사 종류'로만 출력한다. 줄 내용은 출력하지 않는다(데이터 행 노출 방지).
문제가 하나라도 있으면 종료 코드 1이다.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_TOP = ["README.md", "README.ko.md", "requirements.txt", ".gitignore", "LICENSE"]
ALLOWED_DIRS = ["docs", "scripts", "src", "tests"]
EXCLUDED_EXPERIMENTS = {"submission_log"}   # .gitignore와 같게 유지
BANNED_EXT = {".parquet", ".csv", ".zip", ".ipynb", ".npz", ".nc", ".grib", ".grib2", ".grb2", ".pkl",
              ".pptx", ".pdf", ".pages", ".xlsx"}
MAX_BYTES = 1_000_000
ALLOWED_BINARY = {Path("docs/OIBC2026_technical_report.pdf")}   # 집계값만으로 만든 산출물(.gitignore와 같게 유지)
CHECKS = {
    "인증 정보 의심": re.compile(r"(?i)(auth_?key|api_?key|secret|token|password)\s*[=:]\s*['\"][A-Za-z0-9_\-+/]{12,}['\"]"),
    "URL 안 인증키": re.compile(r"(?i)[?&](auth_?key|service_?key|key)=[A-Za-z0-9_\-%]{12,}"),
    "발전소 좌표 의심": re.compile(r"\b3[3-8]\.\d{4,}\b.{0,40}\b12[4-9]\.\d{4,}\b"),
    "발전소 ID": re.compile(r"\bPV\d{3}\b"),
    "관측소 ID": re.compile(r"\bW[AB]\d{3}\b"),
    "로컬 절대 경로": re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
}


def candidates():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split("\n")
        return sorted(ROOT/p for p in out if p and (ROOT/p).exists())   # 작업 트리에서 지운·옮긴 파일 제외
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [ROOT/p for p in ALLOWED_TOP if (ROOT/p).exists()]
        for d in ALLOWED_DIRS:
            files += [p for p in (ROOT/d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        files += sorted(p for p in (ROOT/"experiments").glob("*/metrics.json") if p.parent.name not in EXCLUDED_EXPERIMENTS)
        return sorted(p for p in files if p.name != ".DS_Store")


def main():
    files = candidates()
    if "--list" in sys.argv:
        print("\n".join(str(p.relative_to(ROOT)) for p in files))
        return 0
    issues = []
    for p in files:
        rel = p.relative_to(ROOT)
        if rel in ALLOWED_BINARY:
            if p.stat().st_size > MAX_BYTES:
                issues.append((rel, 0, f"1MB 초과({p.stat().st_size:,} bytes)"))
            continue
        if p.suffix.lower() in BANNED_EXT:
            issues.append((rel, 0, f"금지 확장자 {p.suffix}"))
            continue
        if p.stat().st_size > MAX_BYTES:
            issues.append((rel, 0, f"1MB 초과({p.stat().st_size:,} bytes)"))
        if rel == Path("scripts/check_publish.py"):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            issues.append((rel, 0, "텍스트가 아닌 파일"))
            continue
        for i, line in enumerate(lines, 1):
            for name, rx in CHECKS.items():
                if rx.search(line):
                    issues.append((rel, i, name))
    for rel, i, name in issues:
        print(f"{rel}:{i}  {name}")
    kinds = {}
    for _, _, name in issues:
        kinds[name] = kinds.get(name, 0)+1
    print(f"\n검사 파일 {len(files)}개, 문제 {len(issues)}건 {kinds if kinds else ''}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
