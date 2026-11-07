# CLAUDE.md

프로젝트 설명, 구조, 환경, 규칙은 agent.md에 있습니다.

@agent.md

## Claude Code 작업 방식
- 사용자가 영상 경로를 주며 편집을 부탁하면 `yt-autoedit` 스킬(`~/.claude/skills/yt-autoedit/SKILL.md`)을 따른다. plan.json(일러스트 위치·묘사·자막)은 Claude가 `transcript_cut.txt`를 전부 읽고 쓴다.
- 일러스트는 `run.py sheet`로 모아 직접 보고, 틀린 그림(화살표 방향, 인원수 등)은 프롬프트를 고쳐 `--redo <id> --seed N`으로 다시 그린다.
- 렌더처럼 오래 걸리는 명령은 백그라운드로 돌리고, 시작 전에 예상 시간을 알린다.
- 작업이 끝나면 ComfyUI를 끌지 사용자에게 묻는다(VRAM을 계속 씀).
- 사용자에게는 한국어로, 짧고 쉽게 보고한다.
