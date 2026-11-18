"""Builds notes.md: new v6 section on top, v5 notes moved to history, v6 changelog entry appended."""
prev = open("prev/notes.md", encoding="utf-8-sig").read()
head, rest = prev.split("\n---\n# History: v4 notes", 1)
v5 = head.replace("# Forecast notes: Jul–Oct 2025 per style_color (snapshot 2025-06-30)",
                  "# History: v5 notes (snapshot 2025-06-30), kept for reference")
new = open("research/notes_v6_section.md", encoding="utf-8").read()
changelog_add = open("research/notes_v6_changelog.md", encoding="utf-8").read()
out = new.rstrip("\n") + "\n\n---\n" + v5.rstrip("\n") + "\n\n---\n# History: v4 notes" + rest
out = out.rstrip("\n") + "\n" + changelog_add
open("notes.md", "w", encoding="utf-8").write(out)
print(len(out.splitlines()), "lines")
