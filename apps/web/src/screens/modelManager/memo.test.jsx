import React, { act } from "react";
import { expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { equalItemProps, useStableActions } from "./memo.js";

it("skips unchanged item selections while stable actions use current state", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const render = vi.fn();
  const job = { id: "job" };
  const onAction = vi.fn();
  const Item = React.memo(function Item({ jobs, action }) {
    render();
    return <button onClick={action}>{jobs[0].id}</button>;
  }, equalItemProps);
  function Parent({ value }) {
    const actions = useStableActions({ select: () => onAction(value) });
    return <Item jobs={[job]} action={actions.select} />;
  }
  const { root, container } = mountRoot();
  await act(async () => root.render(<Parent value="before" />));
  await act(async () => root.render(<Parent value="after" />));
  expect(render).toHaveBeenCalledTimes(1);
  await act(async () => container.querySelector("button").click());
  expect(onAction).toHaveBeenCalledWith("after");
  await unmountRoot(root, container);
});
