/* Current-source UI integration and artifact capture. Run with a local Puppeteer installation. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const puppeteer=require('puppeteer-core');
const root=path.resolve(__dirname,'..');

(async()=>{
 const browser=await puppeteer.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,
   args:['--disable-gpu','--no-first-run','--disable-background-networking']});
 const findings={url:process.argv[2]||'http://127.0.0.1:52121',checks:[],page_errors:[]};
 try{
  const page=await browser.newPage();await page.setViewport({width:1600,height:1200,deviceScaleFactor:1});
  page.on('pageerror',e=>findings.page_errors.push(e.message));
  await page.goto(findings.url,{waitUntil:'networkidle0'});
  assert.match(await page.title(),/ETHERGLASS/);
  await page.click('#demo-button');
  await page.waitForFunction(()=>document.querySelector('#capture-stats').textContent.includes('16,384')&&!document.querySelector('#separate-button').disabled,{timeout:90000});
  findings.checks.push('Actual fixed reference loaded:16,384samples');
  await page.click('#separate-button');
  await page.waitForFunction(()=>document.querySelector('[data-trace="target"]').getAttribute('aria-pressed')==='true',{timeout:90000});
  assert.equal(await page.$eval('#classification-source',e=>e.textContent),'RECOVERED TARGET');
  await page.screenshot({path:path.join(root,'assets','observatory.png'),fullPage:true});
  findings.checks.push('Actual trained hybrid reconstruction, synchronized traces and selected-target inference');
  // A deliberately differing inference fixture verifies association, then restore
  // actual response before any screenshots. It never becomes demonstration data.
  const fixture=await page.evaluate(()=>{
   const original=state.analysis;const probe=JSON.parse(JSON.stringify(original));
   probe.classification={label:'fixture-mixture',status:'unknown',confidence:0,reason:'association test'};
   probe.separation.components[0].classification={label:'fixture-target',status:'known',confidence:.91,reason:'association test'};
   receive(probe);document.querySelector('[data-trace="target"]').click();
   const target=document.querySelector('#classification-label').textContent;
   document.querySelector('[data-trace="mixture"]').click();const mixture=document.querySelector('#classification-label').textContent;
   document.querySelector('[data-trace="residual"]').click();const residual=document.querySelector('#classification-label').textContent;
   receive(original);document.querySelector('[data-trace="target"]').click();return {target,mixture,residual};
  });
  assert.deepEqual(fixture,{target:'fixture-target',mixture:'fixture-mixture',residual:'Unassigned residual'});
  findings.checks.push('Mixture/target classification follows trace; residual remains unassigned');
  await page.click('#export-button');await page.waitForSelector('#export-dialog[open]');
  const exportResult=await page.evaluate(async()=>{
    const urls=[...document.querySelectorAll('#export-links a')].map(a=>a.href);
    const statuses=[];for(const url of urls){const r=await fetch(url);statuses.push({status:r.status,bytes:(await r.arrayBuffer()).byteLength,type:r.headers.get('content-type')});}return statuses;
  });
  assert.equal(exportResult.length,3);assert.ok(exportResult.every(x=>x.status===200&&x.bytes>100&&x.type==='application/zip'));
  await page.screenshot({path:path.join(root,'assets','export-dialog.png'),fullPage:false});
  await page.click('#close-export');findings.checks.push('All three actual SigMF+report ZIP exports return200');
  await page.$eval('#start-ms',e=>e.value='2');await page.$eval('#stop-ms',e=>e.value='6');await page.click('#apply-selection');
  await page.waitForFunction(()=>document.querySelector('#capture-stats').textContent.includes('4,000')&&!document.querySelector('#apply-selection').disabled,{timeout:90000});
  assert.equal(await page.evaluate(()=>state.analysis.selection.start_sample),2000);
  assert.equal(await page.evaluate(()=>state.analysis.selection.stop_sample),6000);
  await page.click('#reset-selection');await page.waitForFunction(()=>document.querySelector('#capture-stats').textContent.includes('16,384')&&!document.querySelector('#reset-selection').disabled,{timeout:90000});
  findings.checks.push('Absolute numeric crop and full-source reset recompute correct sample bounds');
  const waterfall=await page.$('#waterfall');await waterfall.scrollIntoView();const rect=await waterfall.boundingBox();
  await page.mouse.move(rect.x+rect.width*.45,rect.y+rect.height*.25);await page.mouse.down();
  await page.mouse.move(rect.x+rect.width*.55,rect.y+rect.height*.65,{steps:8});await page.mouse.up();
  const drag=await page.evaluate(()=>({...state.selection}));assert.ok(drag.start>0&&drag.stop<16384&&drag.stop>drag.start);
  findings.checks.push('Pointer waterfall selection yields valid source sample interval and frequency');
  await page.setViewport({width:390,height:844,deviceScaleFactor:1});await page.evaluate(()=>window.scrollTo(0,0));
  const overflow=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
  assert.ok(overflow.scroll<=overflow.width+1,JSON.stringify(overflow));
  await page.screenshot({path:path.join(root,'assets','observatory-mobile.png'),fullPage:true});
  findings.checks.push('390px mobile layout has no horizontal overflow');
  await page.setViewport({width:1600,height:1200,deviceScaleFactor:1});
  await page.goto(findings.url,{waitUntil:'networkidle0'});
  await (await page.$('#data-file')).uploadFile(path.join(root,'examples/reference/mixture.sigmf-data'));
  await (await page.$('#meta-file')).uploadFile(path.join(root,'examples/reference/mixture.sigmf-meta'));
  const imported=page.waitForResponse(r=>r.url().endsWith('/api/import')&&r.request().method()==='POST');
  await page.click('#import-form button[type="submit"]');
  const importedResponse=await imported;assert.equal(importedResponse.status(),200);
  const importedData=await importedResponse.json();
  await page.waitForFunction(id=>!state.busy&&state.analysis?.capture_id===id,{},importedData.capture_id);
  const referenceManifest=JSON.parse(fs.readFileSync(path.join(root,'examples/reference/manifest.json'),'utf8'));
  assert.equal(importedData.source_sha256,referenceManifest.files['mixture.sigmf-data']);
  assert.equal(importedData.num_samples,16384);
  findings.checks.push('Actual SigMF file/sidecar upload preserves sample count and source SHA256');
  await (await page.$('#data-file')).uploadFile(path.join(root,'examples/real/camras-danuri-65536.sigmf-data'));
  await page.$eval('#meta-file',e=>e.value='');
  await page.$eval('#sample-rate',e=>e.value='500000');
  await page.$eval('#center-frequency',e=>e.value='2260830000');
  await page.select('#datatype','ci16_le');
  const raw=page.waitForResponse(r=>r.url().endsWith('/api/import')&&r.request().method()==='POST');
  await page.click('#import-form button[type="submit"]');
  const rawResponse=await raw;assert.equal(rawResponse.status(),200);
  const rawData=await rawResponse.json();
  await page.waitForFunction(id=>!state.busy&&state.analysis?.capture_id===id,{},rawData.capture_id);
  assert.equal(rawData.num_samples,65536);assert.equal(rawData.sample_rate,500000);
  assert.equal(rawData.center_frequency,2260830000);
  findings.checks.push('Actual raw int16 upload uses explicit rate/frequency/encoding and returns finite measured views');
  const diagram=await browser.newPage();
  await diagram.setViewport({width:1200,height:500,deviceScaleFactor:1});
  await diagram.setContent('<style>html,body{margin:0}</style>'+fs.readFileSync(path.join(root,'assets','architecture.svg'),'utf8'));
  await diagram.screenshot({path:path.join(root,'assets','architecture.png')});
  await diagram.close();
  assert.deepEqual(findings.page_errors,[]);
  findings.ok=true;
 }finally{
  fs.writeFileSync(path.join(root,'research','ui-verification.json'),JSON.stringify(findings,null,2)+'\n');
  await browser.close();
 }
 console.log(JSON.stringify(findings,null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
