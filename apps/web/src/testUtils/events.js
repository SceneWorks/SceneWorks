// Keep listener handles after close: stale-event tests intentionally invoke them to
// prove the production subscription guard ignores callbacks from a closed stream.
export class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.listeners = {};
    this.closed = false;
    FakeEventSource.instances.push(this);
  }

  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }

  close() {
    this.closed = true;
  }
}
