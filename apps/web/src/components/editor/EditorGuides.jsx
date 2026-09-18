import React, { useId, useState } from "react";
import { Modal } from "../Modal.jsx";
import { Markdown } from "../Markdown.jsx";
import { scriptGuide, operatorGuide } from "virtual:editor-guides";

const guides = {
  script: { title: "Write a script & run your first test", content: scriptGuide },
  operator: { title: "Film Editor operator guide", content: operatorGuide },
};
const documentGuides = { "film-script-writing.md": "script", "film-editor.md": "operator" };

// Ship the repository guides with the app so help also works offline. Keep links
// between these two guides in the dialog; technical references open separately.
function guideLinks(content) {
  return content.replace(/\]\(([^)\s]+\.md)\)/g, (_, path) => {
    const guide = documentGuides[path];
    return `](${guide ? `#editor-guide-${guide}` : `https://github.com/SceneWorks/SceneWorks/blob/main/docs/${path}`})`;
  });
}

export function EditorGuides() {
  const titleId = useId();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState("script");
  return (
    <>
      <button className="ve-ghost-btn" aria-haspopup="dialog" onClick={() => setOpen(true)} type="button">Guides</button>
      {open ? (
        <Modal className="prompt-guide-modal editor-guides-modal" labelledBy={titleId} onClose={() => setOpen(false)}>
          <header className="prompt-guide-head">
            <div><p className="eyebrow">Video Editor</p><h2 id={titleId}>Guides</h2></div>
            <button className="modal-close" onClick={() => setOpen(false)} type="button">Close</button>
          </header>
          <nav className="editor-guides-nav" aria-label="Choose a guide">
            {Object.entries(guides).map(([id, guide]) => (
              <button aria-pressed={selected === id} key={id} onClick={() => setSelected(id)} type="button">{guide.title}</button>
            ))}
          </nav>
          <div className="prompt-guide-body" key={selected} onClick={(event) => {
            const link = event.target.closest("a");
            const id = link?.getAttribute("href")?.replace(/^#editor-guide-/, "");
            if (Object.hasOwn(guides, id)) {
              event.preventDefault();
              setSelected(id);
            }
          }}>
            <Markdown content={guideLinks(guides[selected].content)} />
          </div>
        </Modal>
      ) : null}
    </>
  );
}
