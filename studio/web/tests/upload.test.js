import assert from "node:assert/strict";
import { test } from "node:test";
import { uploadJob } from "../src/lib/api.js";

class TestRequest {
  static last;
  constructor() {
    TestRequest.last = this;
    this.upload = {};
    this.headers = {};
  }
  open(method, path) { this.method = method; this.path = path; }
  setRequestHeader(name, value) { this.headers[name] = value; }
  send(body) { this.body = body; }
}

globalThis.XMLHttpRequest = TestRequest;

test("upload reports bytes sent and waits for server acceptance at 100%", async () => {
  const updates = [];
  const form = new FormData();
  const result = uploadJob(form, (value) => updates.push(value));
  const request = TestRequest.last;
  assert.equal(request.method, "POST");
  assert.equal(request.path, "/api/jobs");
  assert.equal(request.body, form);
  assert.deepEqual(request.headers, { "X-UniForge-Studio": "1" });
  request.upload.onprogress({ lengthComputable: true, loaded: 400, total: 1000 });
  request.upload.onload();
  assert.deepEqual(updates, [
    { phase: "uploading", percent: 0 },
    { phase: "uploading", percent: 40 },
    { phase: "checking", percent: 100 },
  ]);
  let accepted = false;
  result.then(() => { accepted = true; });
  await Promise.resolve();
  assert.equal(accepted, false);
  request.status = 202;
  request.response = { id: "accepted-job", status: "queued" };
  request.onload();
  assert.deepEqual(await result, request.response);
});

test("server upload errors retain the actionable error message", async () => {
  const result = uploadJob(new FormData(), () => {});
  const request = TestRequest.last;
  request.status = 413;
  request.response = { detail: "The upload must be smaller than 64 MB." };
  request.onload();
  await assert.rejects(result, /smaller than 64 MB/);
});

for (const [event, message] of [
  ["onerror", /Check your connection/],
  ["ontimeout", /timed out/],
  ["onabort", /cancelled/],
]) {
  test(`upload ${event} releases the failed request with an error`, async () => {
    const result = uploadJob(new FormData(), () => {});
    TestRequest.last[event]();
    await assert.rejects(result, message);
  });
}
