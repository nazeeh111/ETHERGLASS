/* Actual current UI captures, isolated local browser. Requires installed puppeteer-core. */
const puppeteer = require('puppeteer-core');
const path=require('path');
(async()=>{
 const browser=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--disable-gpu','--no-first-run','--disable-background-networking']});
 try {
  const page=await browser.newPage();await page.setViewport({width:1440,height:1050,deviceScaleFactor:1});
  await page.goto(process.argv[2]||'http://127.0.0.1:52121',{waitUntil:'networkidle0'});
  await page.click('#demo-button');await page.waitForFunction(()=>!document.body.matches('[aria-busy="true"]')&&!document.querySelector('#separate-button').disabled);
  const wb=await (await page.$('.workbench')).boundingBox();
  await page.screenshot({path:path.join(__dirname,'../composition/assets/observatory.png'),clip:{x:wb.x,y:wb.y,width:wb.width,height:850}});
  await page.click('#separate-button');await page.waitForFunction(()=>!document.body.matches('[aria-busy="true"]')&&document.querySelector('[data-trace="target"]').getAttribute('aria-pressed')==='true');
  await page.click('#export-button');await page.waitForSelector('#export-dialog[open]');
  await page.$eval('#export-dialog',el=>el.scrollIntoView({block:'center'}));
  const box=await (await page.$('#export-dialog')).boundingBox();
  await page.screenshot({path:path.join(__dirname,'../composition/assets/export.png'),clip:box});
  console.log(JSON.stringify({url:page.url(),title:await page.title(),dialog:await page.$eval('#export-dialog',e=>e.innerText)},null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
