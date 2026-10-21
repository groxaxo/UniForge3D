#!/usr/bin/env node
/* Offline GLB inspection and a four-view PNG. Uses a separate headless browser. */
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { createRequire } = require("node:module");

async function main() {
  const [modelArg, outputArg, titleArg] = process.argv.slice(2);
  if (!modelArg || !outputArg) throw new Error("Usage: node tools/render_mlx_glb.cjs model.glb preview.png [title]");
  const heading = (titleArg || "UniForge3D · local MLX result").replace(/[&<>"']/g,
    character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
  const model = path.resolve(modelArg);
  const output = path.resolve(outputArg);
  const root = path.resolve(__dirname, "..");
  const browserRequire = createRequire(path.join(root, ".deps/render-qa/package.json"));
  const { chromium } = browserRequire("playwright");
  const threeRoot = path.join(root, "studio/web/node_modules/three");
  const html = `<!doctype html><html><head><meta charset="utf-8">
  <style>body{margin:0;background:#f0f2f4;font:18px system-ui;color:#26313d}
  h1{font-size:22px;padding:16px 22px;margin:0}main{display:grid;grid-template-columns:repeat(2,512px)}
  section{position:relative}label{position:absolute;top:12px;left:16px}canvas{display:block}</style>
  <script type="importmap">{"imports":{"three":"/three/build/three.module.js",
  "three/addons/":"/three/examples/jsm/"}}</script></head><body>
  <h1>${heading}</h1><main id="views"></main>
  <script type="module">
  import * as THREE from "three";
  import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
  try {
    const gltf=await new GLTFLoader().loadAsync("/model.glb");
    const box=new THREE.Box3().setFromObject(gltf.scene);
    const center=box.getCenter(new THREE.Vector3()), size=box.getSize(new THREE.Vector3());
    const radius=Math.max(size.x,size.y,size.z);
    if (!Number.isFinite(radius)||radius<=0) throw new Error("Invalid GLB bounds");
    const stats={vertices:0,triangles:0,meshes:0,texturedMeshes:0,bounds:[box.min.toArray(),box.max.toArray()]};
    gltf.scene.traverse(obj=>{
      if(!obj.isMesh)return;
      stats.meshes++; stats.vertices+=obj.geometry.attributes.position.count;
      stats.triangles+=(obj.geometry.index?.count??obj.geometry.attributes.position.count)/3;
      const mats=Array.isArray(obj.material)?obj.material:[obj.material];
      stats.texturedMeshes+=Number(mats.some(mat=>Boolean(mat.map)));
      for(const mat of mats) if(!mat.map){mat.metalness=0;mat.roughness=.75;mat.color?.set("#aeb8c4")}
    });
    const camera=new THREE.PerspectiveCamera(35,1,radius*.01,radius*20);
    const scene=new THREE.Scene();scene.background=new THREE.Color("#edf0f3");
    scene.add(gltf.scene,new THREE.HemisphereLight(0xffffff,0x718096,2.4));
    const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(4,7,5);scene.add(key);
    const fill=new THREE.DirectionalLight(0xffffff,2);fill.position.set(-4,4,-5);scene.add(fill);
    const directions=[["Front",0,.14,2.1],["Three-quarter",1.45,.55,1.55],
                      ["Back",0,.14,-2.1],["Side",-2.1,.25,0]];
    for(const [label,x,y,z] of directions){
      const section=document.createElement("section");
      const text=document.createElement("label");text.textContent=label;section.append(text);
      const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
      renderer.setSize(512,512);renderer.setPixelRatio(1);
      renderer.outputColorSpace=THREE.SRGBColorSpace;
      renderer.toneMapping=THREE.ACESFilmicToneMapping;
      camera.position.copy(center).add(new THREE.Vector3(x,y,z).multiplyScalar(radius));
      camera.lookAt(center);renderer.render(scene,camera);
      section.append(renderer.domElement);document.querySelector("#views").append(section);
    }
    window.__modelStats=stats;window.__renderDone=true;
  }catch(error){window.__renderError=String(error);window.__renderDone=true}
  </script></body></html>`;
  const server = http.createServer((req, res) => {
    if (req.url === "/") { res.setHeader("Content-Type", "text/html"); res.end(html); return; }
    let file;
    if (req.url === "/model.glb") file = model;
    else if (req.url.startsWith("/three/")) {
      file = path.resolve(threeRoot, req.url.slice(7));
      if (!file.startsWith(threeRoot + path.sep)) { res.writeHead(403); res.end(); return; }
    } else { res.writeHead(404); res.end(); return; }
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : "model/gltf-binary");
    fs.createReadStream(file).pipe(res);
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  let browser;
  try {
    browser = await chromium.launch({ channel: "chrome", headless: true,
      args: ["--ignore-gpu-blocklist", "--enable-webgl"] });
    const page = await browser.newPage({ viewport: { width: 1024, height: 1084 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => window.__renderDone, null, { timeout: 120000 });
    const result = await page.evaluate(() => ({ stats: window.__modelStats, error: window.__renderError }));
    if (result.error || errors.length) throw new Error(result.error || errors.join("\n"));
    await page.screenshot({ path: output, fullPage: true, timeout: 120000 });
    fs.writeFileSync(output.replace(/\.png$/, ".render.json"), JSON.stringify(result.stats, null, 2) + "\n");
    console.log(JSON.stringify({ output, ...result.stats }, null, 2));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
