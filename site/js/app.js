import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { MASKS, COSTUMES, COLOR_MEANINGS } from './data.js';

const $ = (s) => document.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'style') el.style.cssText = v;
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
};

const state = {
  mode: 'mask',
  id: 'guanyu',
  face: null, // 戏服模式下的换脸
  gear: true,
  beard: true,
  flags: true,
  rotate: true,
  light: 'stage',
};

// ------------------------------------------------------------ three.js 舞台
const canvas = $('#view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0f);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 60);
camera.position.set(-0.4, 0.1, 1.2);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.7;
controls.minDistance = 0.25;
controls.maxDistance = 9;

// 灯光
const key = new THREE.SpotLight(0xfff1de, 0, 0, Math.PI / 7, 0.6, 1.2);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.bias = -0.0002;
key.shadow.normalBias = 0.01;
const rim = new THREE.DirectionalLight(0xbcd4ff, 0);
const fill = new THREE.HemisphereLight(0xfff6ea, 0x1a1410, 0);
scene.add(key, key.target, rim, rim.target, fill);

function applyLight() {
  const r = currentRadius;
  if (state.light === 'stage') {
    scene.environmentIntensity = 0.45;
    key.intensity = 38 * r * r;
    rim.intensity = 2.2;
    fill.intensity = 0.25;
    scene.background.set(0x0b0b0d);
  } else {
    scene.environmentIntensity = 1.0;
    key.intensity = 16 * r * r;
    rim.intensity = 0.9;
    fill.intensity = 0.7;
    scene.background.set(0x1a1a1d);
  }
}

// 地面阴影 + 头像底座
const floor = new THREE.Mesh(new THREE.CircleGeometry(4, 64), new THREE.ShadowMaterial({ opacity: 0.45 }));
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const pedestal = new THREE.Group();
{
  const lacquer = new THREE.MeshPhysicalMaterial({ color: 0x101010, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.08 });
  const gold = new THREE.MeshStandardMaterial({ color: 0xd9a74a, metalness: 1, roughness: 0.3 });
  const col = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.07, 0.2, 64), lacquer);
  col.position.y = -0.215 - 0.1;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.059, 0.004, 16, 80), gold);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -0.218;
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.14, 0.03, 80), lacquer);
  base.position.y = -0.215 - 0.2 - 0.015;
  const ring2 = new THREE.Mesh(new THREE.TorusGeometry(0.131, 0.003, 16, 120), gold);
  ring2.rotation.x = Math.PI / 2;
  ring2.position.y = -0.215 - 0.2;
  for (const m of [col, base, ring, ring2]) { m.castShadow = true; m.receiveShadow = true; pedestal.add(m); }
}
scene.add(pedestal);

const holder = new THREE.Group();
scene.add(holder);

// ------------------------------------------------------------ 模型加载
const draco = new DRACOLoader().setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/libs/draco/gltf/');
const loader = new GLTFLoader().setDRACOLoader(draco);
const cache = new Map();
const HAIR_COLORS = { Hair_black: 0x0b0a0a, Hair_red: 0x8c0f0c, Hair_white: 0xe6e1d6, Hair_yellow: 0xe0b23a, Hair_redtassel: 0xa30d14, Hair_pink: 0xe07a90, Hair_gold: 0xd9a030, Hair_green: 0x1f7a44 };

function prepare(root) {
  root.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    o.receiveShadow = true;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (m.userData.prepped) continue;
      m.userData.prepped = true;
      if (/^(Hair_|Feather)/.test(m.name)) {
        m.transparent = false;
        m.alphaTest = 0.35;
        m.alphaToCoverage = true;
        m.depthWrite = true;
        m.side = THREE.DoubleSide;
        const base = m.name.split('.')[0];
        if (HAIR_COLORS[base] !== undefined) m.color.setHex(HAIR_COLORS[base]);
        o.castShadow = false;
      }
      if (/^(Cloth_|Satin|Skirt)/.test(m.name)) m.side = THREE.DoubleSide;
      m.needsUpdate = true;
    }
  });
  return root;
}

function load(url) {
  if (!cache.has(url)) {
    cache.set(url, new Promise((res, rej) => loader.load(url, (g) => res(prepare(g.scene)), (e) => onProgress(e), rej)));
  }
  return cache.get(url).then((s) => s.clone(true));
}

let loadingTimer = 0;
function onProgress(e) {
  if (e.lengthComputable) $('#loadingText').textContent = `正在加载模型… ${Math.round((e.loaded / e.total) * 100)}%`;
}
function setLoading(on) {
  clearTimeout(loadingTimer);
  if (on) loadingTimer = setTimeout(() => $('#loading').classList.add('show'), 120);
  else { $('#loading').classList.remove('show'); $('#loadingText').textContent = '正在加载模型…'; }
}

// ------------------------------------------------------------ 取景
let homeView = null;
let currentRadius = 1;

function boxOf(obj, skip = /Lingzi|Kaoqi|Diao_Wing|Diao_Rod/) {
  const box = new THREE.Box3();
  obj.updateWorldMatrix(true, true);
  obj.traverse((o) => {
    if (!o.isMesh || !o.visible) return;
    let p = o, hidden = false, skipped = false;
    while (p) { if (!p.visible) hidden = true; if (skip.test(p.name)) skipped = true; p = p.parent; }
    if (hidden || skipped) return;
    box.expandByObject(o);
  });
  return box;
}

function frame(box, dir, pad = 1.12, targetShift = 0) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  center.y += targetShift;
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = Math.max(size.y / 2 / Math.tan(vfov / 2), Math.max(size.x, size.z) / 2 / Math.tan(hfov / 2)) * pad + size.z / 2;
  const d = dir.clone().normalize();
  homeView = { pos: center.clone().addScaledVector(d, dist), target: center.clone() };
  camera.position.copy(homeView.pos);
  controls.target.copy(homeView.target);
  camera.near = dist / 100;
  camera.far = dist * 20;
  camera.updateProjectionMatrix();
  controls.minDistance = dist * 0.25;
  controls.maxDistance = dist * 3;
  // 灯光跟随尺度
  currentRadius = Math.max(size.y, size.x) * 0.5 + 0.2;
  const R = dist * 1.4;
  key.position.set(center.x - R * 0.55, center.y + R * 0.9, center.z + R * 0.75);
  key.target.position.copy(center);
  key.distance = 0;
  key.shadow.camera.near = R * 0.3;
  key.shadow.camera.far = R * 3;
  key.shadow.camera.updateProjectionMatrix();
  rim.position.set(center.x + R * 0.6, center.y + R * 0.6, center.z - R);
  rim.target.position.copy(center);
  applyLight();
}

// ------------------------------------------------------------ 切换展示
let token = 0;
async function show() {
  const my = ++token;
  setLoading(true);
  let group;
  try {
    if (state.mode === 'mask') {
      group = await load(`models/head_${state.id}.glb`);
    } else {
      const c = COSTUMES.find((x) => x.id === state.id);
      const face = state.face || c.head;
      const [body, head] = await Promise.all([load(`models/costume_${state.id}.glb`), load(`models/head_${face}.glb`)]);
      group = body;
      const socket = body.getObjectByName('HeadSocket');
      (socket || body).add(head);
    }
  } catch (err) {
    console.error(err);
    if (my === token) { $('#loadingText').textContent = '模型加载失败，请刷新重试'; }
    return;
  }
  if (my !== token) return;
  holder.clear();
  holder.add(group);
  pedestal.visible = state.mode === 'mask';
  floor.position.y = state.mode === 'mask' ? -0.215 - 0.23 : 0;
  applyVisibility();
  if (state.mode === 'mask') {
    const b = boxOf(group);
    b.min.y = Math.max(b.min.y, -0.5);
    frame(b, new THREE.Vector3(-0.32, 0.08, 1), 1.1);
  } else {
    frame(boxOf(group, /Lingzi/), new THREE.Vector3(-0.35, 0.12, 1), 1.06);
  }
  setLoading(false);
}

function nodes(re) {
  const out = [];
  holder.traverse((o) => { if (re.test(o.name)) out.push(o); });
  return out;
}

function applyVisibility() {
  for (const n of nodes(/^Headgear/)) n.visible = state.gear;
  for (const n of nodes(/^Beard/)) n.visible = state.beard;
  for (const n of nodes(/^Kaoqi/)) n.visible = state.flags;
}

// ------------------------------------------------------------ 界面
const HERO = {
  mask: { kicker: 'LIAN PU · 脸谱', title: '一张脸，就是一个<span class="hl">性格</span>', lede: '拖动旋转、滚轮缩放。每一张脸谱都绘制在真实比例的头部模型上，配好盔头与髯口，可以随时摘下来细看。', gal: '八张脸谱', galK: 'MASKS' },
  costume: { kicker: 'XI FU · 戏服', title: '宁穿破，不穿<span class="hl">错</span>', lede: '蟒、靠、女蟒、猴衣——每一件都按穿戴规矩搭好盔头与髯口。还可以给戏服“换脸”，看看张飞穿上绿蟒是什么样子。', gal: '五套戏服', galK: 'COSTUMES' },
};

function renderHero() {
  const c = HERO[state.mode];
  $('#kicker').textContent = c.kicker;
  $('#heroTitle').innerHTML = c.title;
  $('#heroLede').textContent = c.lede;
  $('#galTitle').textContent = c.gal;
  $('#galKicker').textContent = c.galK;
  document.body.classList.toggle('costume-mode', state.mode === 'costume');
  $('#meaning').style.display = state.mode === 'mask' ? '' : 'none';
  for (const b of document.querySelectorAll('#modeSeg button')) b.classList.toggle('active', b.dataset.mode === state.mode);
}

function renderGrid() {
  const grid = $('#grid');
  grid.innerHTML = '';
  const list = state.mode === 'mask' ? MASKS : COSTUMES;
  for (const it of list) {
    const img = h('img', { src: `thumbs/${state.mode === 'mask' ? 'head' : 'costume'}_${it.id}.webp`, alt: it.name, loading: 'lazy' });
    img.onerror = () => { img.onerror = null; img.src = img.src.replace('.webp', '.png'); };
    const tile = h('button', { class: 'tile' + (it.id === state.id ? ' active' : ''), onclick: () => select(it.id) },
      h('div', { class: 'tile-img' }, img),
      h('div', { class: 'tile-meta' },
        h('div', { class: 'tile-name' }, state.mode === 'mask' ? it.name : `${it.name} · ${it.wearer}`),
        h('div', { class: 'tile-sub' }, state.mode === 'mask' ? it.type : it.category)));
    grid.append(tile);
  }
}

function renderInfo() {
  const info = $('#info');
  info.style.animation = 'none';
  void info.offsetWidth;
  info.style.animation = '';
  const facts = $('#facts');
  facts.innerHTML = '';
  const row = (k, v) => facts.append(h('dt', {}, k), h('dd', {}, v));
  const tags = (arr) => h('div', { class: 'tags' }, arr.map((t) => h('span', { class: 'tag' }, t)));
  if (state.mode === 'mask') {
    const m = MASKS.find((x) => x.id === state.id);
    $('#infoSwatch').style.background = m.swatch;
    $('#infoType').textContent = m.type;
    $('#infoName').textContent = m.name;
    row('行当', m.role);
    row('主色', h('div', { class: 'tags' }, m.colors.map((c) => h('span', { class: 'tag' }, c))));
    row('穿戴', m.gear);
    row('剧目', tags(m.plays.map((p) => `《${p}》`)));
    $('#infoDesc').textContent = m.desc;
  } else {
    const c = COSTUMES.find((x) => x.id === state.id);
    $('#infoSwatch').style.background = c.swatch;
    $('#infoType').textContent = c.category;
    $('#infoName').textContent = `${c.name} · ${c.wearer}`;
    row('构件', tags(c.parts));
    row('剧目', tags(c.plays.map((p) => `《${p}》`)));
    $('#infoDesc').textContent = c.desc;
  }
  renderToggles();
}

function sw(label, prop) {
  const b = h('button', { class: 'switch' + (state[prop] ? ' on' : ''), role: 'switch', 'aria-checked': String(state[prop]), 'aria-label': label });
  b.onclick = () => { state[prop] = !state[prop]; b.classList.toggle('on', state[prop]); b.setAttribute('aria-checked', String(state[prop])); applyVisibility(); };
  return h('div', { class: 'toggle' }, h('span', {}, label), b);
}

function renderToggles() {
  const t = $('#toggles');
  t.innerHTML = '';
  const noBeard = ['sunwukong', 'yangguifei'];
  if (state.mode === 'mask') {
    t.append(sw('盔头 / 头饰', 'gear'));
    if (!noBeard.includes(state.id)) t.append(sw('髯口', 'beard'));
  } else {
    const c = COSTUMES.find((x) => x.id === state.id);
    const sel = h('select', { class: 'select', 'aria-label': '换脸' }, MASKS.map((m) => {
      const o = h('option', { value: m.id }, `${m.name} · ${m.type}`);
      if ((state.face || c.head) === m.id) o.selected = true;
      return o;
    }));
    sel.onchange = () => { state.face = sel.value === c.head ? null : sel.value; show(); syncHash(); };
    t.append(h('div', { class: 'toggle' }, h('span', {}, '换脸'), sel));
    t.append(sw('盔头 / 头饰', 'gear'));
    t.append(sw('髯口', 'beard'));
    if (state.id === 'xiangyu') t.append(sw('靠旗', 'flags'));
  }
}

function renderColors() {
  $('#colors').append(...COLOR_MEANINGS.map((c) => h('div', { class: 'color-card' },
    h('div', { class: 'color-chip', style: `background:${c.c}` }),
    h('div', { class: 'color-name' }, c.n),
    h('div', { class: 'color-mean' }, c.m),
    h('div', { class: 'color-eg' }, `例：${c.e}`))));
}

function select(id) {
  if (id === state.id) return;
  state.id = id;
  state.face = null;
  for (const el of document.querySelectorAll('.tile')) el.classList.remove('active');
  const list = state.mode === 'mask' ? MASKS : COSTUMES;
  document.querySelectorAll('.tile')[list.findIndex((x) => x.id === id)]?.classList.add('active');
  renderInfo();
  show();
  syncHash();
  if (innerWidth < 960) $('#stage').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  const list = mode === 'mask' ? MASKS : COSTUMES;
  if (!list.some((x) => x.id === state.id)) state.id = list[0].id;
  state.face = null;
  renderHero();
  renderGrid();
  renderInfo();
  show();
  syncHash();
}

function syncHash() {
  const f = state.mode === 'costume' && state.face ? `/${state.face}` : '';
  history.replaceState(null, '', `#${state.mode}/${state.id}${f}`);
}

function readHash() {
  const [mode, id, face] = location.hash.replace('#', '').split('/');
  if (mode === 'mask' || mode === 'costume') state.mode = mode;
  const list = state.mode === 'mask' ? MASKS : COSTUMES;
  if (list.some((x) => x.id === id)) state.id = id;
  if (state.mode === 'costume' && MASKS.some((m) => m.id === face)) state.face = face;
}

// 控件
for (const b of document.querySelectorAll('#modeSeg button')) b.onclick = () => setMode(b.dataset.mode);
for (const b of document.querySelectorAll('#lightSeg button')) b.onclick = () => {
  state.light = b.dataset.light;
  for (const x of document.querySelectorAll('#lightSeg button')) x.classList.toggle('active', x === b);
  applyLight();
};
$('#rotBtn').onclick = () => { state.rotate = !state.rotate; controls.autoRotate = state.rotate; $('#rotBtn').classList.toggle('active', state.rotate); };
$('#resetBtn').onclick = () => { if (!homeView) return; camera.position.copy(homeView.pos); controls.target.copy(homeView.target); };
$('#shotBtn').onclick = () => {
  renderer.render(scene, camera);
  const name = state.mode === 'mask' ? MASKS.find((m) => m.id === state.id).name : COSTUMES.find((m) => m.id === state.id).wearer;
  const a = h('a', { download: `京剧-${name}.png`, href: canvas.toDataURL('image/png') });
  a.click();
};
$('#themeBtn').onclick = () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) {}
};
controls.addEventListener('start', () => { controls.autoRotate = false; });
controls.addEventListener('end', () => { setTimeout(() => { controls.autoRotate = state.rotate; }, 2500); });

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

addEventListener('hashchange', () => {
  const before = `${state.mode}/${state.id}/${state.face}`;
  readHash();
  if (`${state.mode}/${state.id}/${state.face}` === before) return;
  renderHero();
  renderGrid();
  renderInfo();
  show();
});

readHash();
renderHero();
renderGrid();
renderInfo();
renderColors();
resize();
show();

// 空闲时预取其它模型
const idle = window.requestIdleCallback || ((f) => setTimeout(f, 1500));
idle(() => {
  for (const m of MASKS) load(`models/head_${m.id}.glb`).catch(() => {});
});
