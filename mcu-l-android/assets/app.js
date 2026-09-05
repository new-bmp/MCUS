(function(){
  'use strict';
  const catalog=window.MCU_CATALOG;
  if(!catalog){document.querySelector('.splash-sub').textContent='目录载入失败';return;}
  const $=s=>document.querySelector(s);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const value=v=>v===undefined||v===null||v===''?'—':v;
  // Keep the three states distinct in engineering views: a documented zero is
  // different from a field that was never verified in the source data.
  const engineeringValue=v=>{
    if(v===undefined||v===null||v===''||String(v).toLowerCase()==='unknown'||String(v).toLowerCase()==='not_found')return '未核验';
    if(v===0||String(v)==='0')return '无';
    return v;
  };
  const engineeringMemory=v=>{
    if(v===undefined||v===null||v===''||String(v).toLowerCase()==='unknown')return '未核验';
    if(Number(v)===0)return '无';
    return memory(v);
  };
  const yesNoValue=v=>v==='yes'?'有':v==='no'?'无':'未核验';
  const count=v=>Number(v||0).toLocaleString('zh-CN');
  const clock=v=>!v?'—':v>=1e9?(v/1e9).toFixed(v%1e9?2:0)+' GHz':v>=1e6?(v/1e6).toFixed(v%1e6?1:0)+' MHz':v>=1e3?(v/1e3).toFixed(0)+' kHz':v+' Hz';
  const memory=v=>!v?'—':v>=1048576?(v/1048576).toFixed(v%1048576?1:0)+' MB':v>=1024?(v/1024).toFixed(v%1024?1:0)+' KB':v+' B';
  const natural=(a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true,sensitivity:'base'});
  const devices=catalog.devices;
  const byId=new Map(devices.map(d=>[d.id,d]));
  const manufacturers=[...new Set(devices.map(d=>d.m))].sort(natural);
  const cores=[...new Set(devices.map(d=>d.c||d.a).filter(Boolean))].sort(natural);
  const coverage=new Map(catalog.coverage.map(v=>[v.m,v]));
  const localModel=window.MCUS_LOCAL_MODEL||{name:'规则引擎',version:'fallback',vendors:[],cores:[],features:[]};
  const quotesEnabled=window.MCUS_QUOTES_ENABLED===true;
  const peripheralFilters=[
    {key:'tim',label:'TIM / 定时器',aliases:'tim timer 定时器 计数器'},
    {key:'pwm',label:'PWM',aliases:'pwm 脉宽调制 电机控制'},
    {key:'adch',label:'ADC 通道',aliases:'adc 模数转换 模拟通道'},
    {key:'adcu',label:'ADC 转换器单元',aliases:'adc converter 模数转换器'},
    {key:'dac',label:'DAC',aliases:'dac 数模转换'},
    {key:'gpio',label:'GPIO',aliases:'gpio io 输入输出'},
    {key:'uart',label:'UART',aliases:'uart 串口 异步串口'},
    {key:'usart',label:'USART',aliases:'usart 串口 同步异步串口'},
    {key:'sercom',label:'SERCOM / FLEXCOM',aliases:'sercom flexcom 可配置串行'},
    {key:'spi',label:'SPI',aliases:'spi 串行外设接口'},
    {key:'i2c',label:'I²C',aliases:'i2c i²c 两线总线'},
    {key:'i2s',label:'I²S',aliases:'i2s i²s 音频接口'},
    {key:'can',label:'CAN / TWAI',aliases:'can canfd can-fd twai 总线'},
    {key:'usb',label:'USB（角色未标明）',aliases:'usb usb hs usb fs 通用usb'},
    {key:'usbd',label:'USB Device',aliases:'usb device usb设备'},
    {key:'usbh',label:'USB Host',aliases:'usb host usb主机'},
    {key:'otg',label:'USB OTG',aliases:'usb otg'},
    {key:'eth',label:'Ethernet',aliases:'ethernet eth 以太网'},
    {key:'sdio',label:'SDIO / SDMMC',aliases:'sdio sdmmc sd卡'},
    {key:'dma',label:'DMA',aliases:'dma 直接存储器访问'},
    {key:'wdt',label:'看门狗',aliases:'watchdog wdt iwdg wwdg 看门狗'},
    {key:'rtc',label:'RTC',aliases:'rtc 实时时钟'},
    {key:'rng',label:'硬件随机数',aliases:'rng trng random 随机数'},
    {key:'comp',label:'比较器',aliases:'comparator comp 比较器'},
    {key:'opamp',label:'运算放大器',aliases:'opamp op amp 运算放大器'},
    {key:'touch',label:'触摸感应',aliases:'touch capacitive 触摸 电容感应'},
    {key:'cam',label:'摄像头接口',aliases:'camera dcmi dvp 摄像头 相机'},
    {key:'display',label:'显示控制器',aliases:'display lcd glcd 显示控制器'},
    {key:'extbus',label:'外部存储总线',aliases:'external bus fmc fsmc qspi octospi 外部存储总线'},
    {key:'tempsens',label:'温度传感器',aliases:'temperature sensor temp 温度传感器'},
    {key:'crypto',label:'硬件加密',aliases:'crypto aes hash sha pka 加密 安全'},
    {key:'wifi',label:'Wi-Fi',aliases:'wifi wi-fi 无线局域网'},
    {key:'bluetooth',label:'Bluetooth',aliases:'bluetooth ble 蓝牙'}
  ];
  const peripheralByKey=new Map(peripheralFilters.map(item=>[item.key,item]));
  function inventoryPresence(d,...types){const wanted=new Set(types.flat().map(v=>String(v).toLowerCase()));return (d.pi||[]).some(item=>wanted.has(String(item.t||'').toLowerCase()))?1:null}
  function peripheralCount(d,key){
    const direct={tim:'tim',pwm:'pwm',adch:'adch',adcu:'adcu',dac:'dac',gpio:'gpio',uart:'uart',usart:'usart',sercom:'sercom',spi:'spi',i2c:'i2c',i2s:'i2s',can:'can',usb:'usb',usbd:'usbd',usbh:'usbh',otg:'otg',eth:'eth',sdio:'sdio',dma:'dma',wdt:'wdt',comp:'comp',opamp:'opamp',touch:'touch',cam:'cam',display:'display',extbus:'extbus',tempsens:'tempsens'};
    if(direct[key]){const v=d[direct[key]];return typeof v==='number'&&Number.isFinite(v)&&v>0?v:null}
    if(key==='rtc')return d.rtc==='yes'?1:inventoryPresence(d,'RTC');
    if(key==='rng')return inventoryPresence(d,'RNG');
    if(key==='crypto')return d.crypto==='yes'?1:inventoryPresence(d,'Crypto');
    if(key==='wifi')return inventoryPresence(d,'WiFi','WiFi6');
    if(key==='bluetooth')return inventoryPresence(d,'Bluetooth');
    return null;
  }
  devices.forEach(d=>{const peripheralText=(d.pi||[]).flatMap(item=>[item.n,item.t,item.d]).join(' ');const aliases=peripheralFilters.filter(item=>peripheralCount(d,item.key)).map(item=>item.aliases).join(' ');const vendorAliases=d.m==='Qinheng'?'沁恒 wch qinheng nanjing qinheng microelectronics qingke 青稞':d.m==='STC'?'stc 宏晶 hongjing stc microelectronics 8051':d.m==='HPMicro'?'先楫 hpm hpmicro risc-v 上海先楫':d.m==='Renesas'?'瑞萨 renesas ra rx rl78 rh850 synergy risc-v 瑞萨电子':d.m==='Artery'?'雅特力 artery arterytek at32 at32f at32a at32l at32m at32wb':d.m==='Allwinner'?'全志 allwinner xradio 芯之联 wireless mcu 实时 异构 soc 数传':d.m==='MicroPy MCU'?'micropy micropython mpy raspberry pi rp2040 rp2350 kendryte k210 micropython mcu':d.m==='Texas Instruments'?'德州仪器 texas instruments ti c2000 tms320 c28x dsp 实时控制器':' ';d._q=[d.n,d.l,d.s,d.f,d.m,d.pt,vendorAliases,d.v,d.c,d.a,peripheralText,aliases,...(d.boards||[]),...(d.parts||[]).map(p=>p.n)].join(' ').toLowerCase()});
  function readStoredArray(key){
    try{const parsed=JSON.parse(localStorage.getItem(key)||'[]');return Array.isArray(parsed)?parsed:[]}
    catch(_){try{localStorage.removeItem(key)}catch(__){}return []}
  }
  function writeStored(key,value){try{localStorage.setItem(key,JSON.stringify(value));return true}catch(_){return false}}
  function installKeyboardViewport(){
    const root=document.documentElement;
    const view=$('#view');
    let resizeTimer;
    const update=()=>{
      const viewport=window.visualViewport;
      const layoutHeight=Math.max(document.documentElement.clientHeight||0,window.innerHeight||0);
      const visibleHeight=viewport?viewport.height:layoutHeight;
      const offsetTop=viewport?viewport.offsetTop:0;
      const inset=Math.max(0,Math.round(layoutHeight-visibleHeight-offsetTop));
      root.style.setProperty('--keyboard-inset',`${inset}px`);
      root.classList.toggle('keyboard-visible',inset>0);
      if(inset>0)ensureAssistantInputVisible();
    };
    const delayedUpdate=()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(update,40)};
    const ensureAssistantInputVisible=()=>{
      const input=$('#assistant-input');
      if(!input||document.activeElement!==input)return;
      const viewport=window.visualViewport;
      const keyboardTop=(viewport?viewport.offsetTop:0)+(viewport?viewport.height:window.innerHeight);
      const rect=input.getBoundingClientRect();
      const margin=12;
      if(rect.bottom>keyboardTop-margin&&view){view.scrollTop+=rect.bottom-(keyboardTop-margin)}
      const messages=$('#assistant-messages');
      if(messages)messages.scrollTop=messages.scrollHeight;
    };
    window.addEventListener('resize',delayedUpdate,{passive:true});
    window.addEventListener('orientationchange',delayedUpdate,{passive:true});
    if(window.visualViewport){window.visualViewport.addEventListener('resize',delayedUpdate,{passive:true});window.visualViewport.addEventListener('scroll',delayedUpdate,{passive:true})}
    document.addEventListener('focusin',event=>{if(event.target&&event.target.id==='assistant-input'){delayedUpdate();setTimeout(ensureAssistantInputVisible,180)}},{passive:true});
    document.addEventListener('focusout',event=>{if(event.target&&event.target.id==='assistant-input'){setTimeout(update,120)}},{passive:true});
    update();
  }
  const state={tab:'catalog',query:'',vendorFilter:'',coreFilter:'',peripheralFilter:'',peripheralMin:1,sort:'score',limit:120,detail:null,browse:{vendor:null,series:null,line:null},compare:new Set(readStoredArray('mcul_compare').filter(id=>byId.has(id))),assistantMessages:[]};
  const previewDevice=new URLSearchParams(location.search).get('device');
  let toastTimer,searchTimer,quoteAbort;

  function group(list,key){const map=new Map();list.forEach(item=>{const k=typeof key==='function'?key(item):item[key];if(!map.has(k))map.set(k,[]);map.get(k).push(item)});return map}
  function unique(list,key){return new Set(list.map(item=>item[key]).filter(Boolean)).size}
  function partCount(list){return list.reduce((sum,d)=>sum+(d.parts||[]).length,0)}
  function summary(label,val){return `<div class="summary-item"><b>${esc(val)}</b><span>${esc(label)}</span></div>`}
  function toast(message){const el=$('#toast');el.textContent=message;el.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),1500)}
  function saveCompare(){writeStored('mcul_compare',[...state.compare]);updateNav()}
  function setTab(tab){if(searchTimer){clearTimeout(searchTimer);searchTimer=null}state.tab=tab;state.detail=null;$('#view').scrollTop=0;render()}
  function scheduleSearchRender(){if(searchTimer)clearTimeout(searchTimer);searchTimer=setTimeout(()=>{searchTimer=null;renderSearch(false)},80)}
  function updateNav(){document.querySelectorAll('#bottom-nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));const badge=$('#compare-badge');badge.textContent=state.compare.size;badge.classList.toggle('show',state.compare.size>0)}
  function renderHeader(){
    const searchable=state.tab==='search';
    document.documentElement.style.setProperty('--fl-top',searchable?'108px':'58px');
    $('#search-slot').innerHTML=searchable?`<div class="search-box"><span>⌕</span><input id="search" value="${esc(state.query)}" placeholder="搜索型号、订货号或外设"><button id="clear-search">${state.query?'×':'↵'}</button></div>`:'';
    if(searchable){const input=$('#search');input.addEventListener('input',e=>{state.query=e.target.value;state.limit=120;scheduleSearchRender()});$('#clear-search').onclick=()=>{if(state.query){state.query='';input.value='';if(searchTimer){clearTimeout(searchTimer);searchTimer=null}renderSearch(false)}else input.focus()}}
  }
  function breadcrumb(items){return `<div class="breadcrumb">${items.map((item,i)=>`${i?'<span>›</span>':''}<button data-crumb="${item.level}">${esc(item.label)}</button>`).join('')}</div>`}
  function vendorGlyph(name){if(name==='STMicroelectronics')return 'ST';if(name==='Texas Instruments')return 'TI';if(name==='Qinheng')return 'WCH';if(name==='HPMicro')return 'HPM';if(name==='Artery')return 'AT';if(name==='Renesas')return 'RE';return name.replace(/[^A-Z]/g,'').slice(0,2)||name.slice(0,2).toUpperCase()}
  const vendorLogoFiles={Allwinner:'allwinner.png',Artery:'artery.svg',Espressif:'espressif.svg',Geehy:'geehy.ico',GigaDevice:'gigadevice.svg',HPMicro:'hpmicro.png',Infineon:'infineon.svg',Microchip:'microchip.ico',MindMotion:'mindmotion.png',Nuvoton:'nuvoton.jpg',Puya:'puya.ico',Qinheng:'qinheng.svg',Renesas:'renesas.svg',STC:'stc.svg',STMicroelectronics:'stmicroelectronics.svg','Texas Instruments':'texas-instruments.ico'};
  function vendorLogo(name){const file=vendorLogoFiles[name];const cssName=name.toLowerCase().replace(/[^a-z0-9]+/g,'-');return `<span class="folder-icon vendor logo-${cssName}"><span class="vendor-fallback">${esc(vendorGlyph(name))}</span>${file?`<img src="vendor-${esc(file)}" alt="${esc(name)} Logo" onerror="this.remove()">`:''}</span>`}
  function vendorName(name){if(name==='Allwinner')return '全志（Allwinner / XRadio）';if(name==='Artery')return '雅特力（Artery / ArteryTek）';if(name==='Microchip')return 'Microchip（原 Atmel）';if(name==='Qinheng')return '沁恒（WCH）';if(name==='Renesas')return '瑞萨电子（Renesas）';if(name==='STC')return 'STC（宏晶）';if(name==='HPMicro')return '先楫半导体（HPMicro）';return name}
  function productType(type){return ({wireless_mcu:'无线 MCU',wireless_audio_mcu_soc:'无线音频 MCU SoC',wireless_connectivity_chip:'无线连接芯片',heterogeneous_realtime_soc:'带实时 MCU 核的 SoC',micropython_mcu:'MicroPython MCU',dsp_mcu:'DSP 实时控制器'})[type]||'MCU'}
  function categoryTitle(list){if(list[0]?.m==='Allwinner'||list[0]?.m==='MicroPy MCU')return [...new Set(list.map(d=>productType(d.pt)))].join(' / ');return list[0]?.a||list[0]?.c||'MCU 系列'}
  function boardTags(d,full=false){const boards=d.boards||[];if(!boards.length)return '';const shown=full?boards:boards.slice(0,3);return `<div class="board-tags"><span>Arduino 开发板</span>${shown.map(name=>`<i>${esc(name)}</i>`).join('')}${!full&&boards.length>shown.length?`<i>+${boards.length-shown.length}</i>`:''}</div>`}
  function maxClock(list){return Math.max(0,...list.map(d=>d.hz||0))}
  function renderCatalog(){
    const browse=state.browse;
    if(!browse.vendor){
      $('#view').innerHTML=`<div class="page-heading"><h1>芯片目录</h1></div><div class="summary-strip">${summary('厂商',catalog.meta.manufacturers)}${summary('系列大类',count(catalog.meta.series))}${summary('器件变体',count(catalog.meta.devices))}</div><div class="section-heading"><h2>厂商</h2><span>${manufacturers.length} 家</span></div><div class="folder-list">${manufacturers.map(m=>{const c=coverage.get(m)||{};return `<button class="folder-row" data-vendor="${esc(m)}">${vendorLogo(m)}<span class="folder-main"><h3>${esc(vendorName(m))}</h3><p>${count(c.series)} 个系列大类 · ${count(c.lines)} 条产品线</p></span><span class="folder-meta"><b>${count(c.devices)}</b><span>器件变体 <i class="folder-chevron">›</i></span></span></button>`}).join('')}</div>`;
      document.querySelectorAll('[data-vendor]').forEach(b=>b.onclick=()=>{browse.vendor=b.dataset.vendor;browse.series=null;browse.line=null;$('#view').scrollTop=0;renderCatalog()});
      return;
    }

    const vendorDevices=devices.filter(d=>d.m===browse.vendor);
    if(!browse.series){
      const categories=[...group(vendorDevices,'s').entries()].sort((a,b)=>natural(a[0],b[0]));
      $('#view').innerHTML=`${breadcrumb([{level:'root',label:'芯片目录'},{level:'vendor',label:browse.vendor}])}<div class="page-heading"><h1>${esc(browse.vendor)}</h1><p>先按厂商定义的芯片系列大类进入，再选择产品线和具体器件变体。</p></div><div class="summary-strip">${summary('系列大类',categories.length)}${summary('产品线',unique(vendorDevices,'l'))}${summary('器件变体',count(vendorDevices.length))}</div><div class="section-heading"><h2>系列大类</h2><span>第 1 层</span></div><div class="category-grid">${categories.map(([series,list])=>`<button class="category-tile" data-series="${esc(series)}"><div class="category-code">${esc(series)}</div><div class="category-title">${esc(categoryTitle(list))}</div><div class="category-stats">${unique(list,'l')} 条产品线 · ${count(list.length)} 个变体<br>最高 ${clock(maxClock(list))} · ${partCount(list)} 个订货号</div></button>`).join('')}</div>`;
      bindCrumbs();document.querySelectorAll('[data-series]').forEach(b=>b.onclick=()=>{browse.series=b.dataset.series;browse.line=null;$('#view').scrollTop=0;renderCatalog()});return;
    }

    const seriesDevices=vendorDevices.filter(d=>d.s===browse.series);
    if(!browse.line){
      const lines=[...group(seriesDevices,'l').entries()].sort((a,b)=>natural(a[0],b[0]));
      $('#view').innerHTML=`${breadcrumb([{level:'root',label:'芯片目录'},{level:'vendor',label:browse.vendor},{level:'series',label:browse.series}])}<div class="page-heading"><h1>${esc(browse.series)}</h1><p>选择该系列下的产品线，再查看封装、存储和版本后缀对应的具体变体。</p></div><div class="summary-strip">${summary('产品线',lines.length)}${summary('器件变体',count(seriesDevices.length))}${summary('完整订货号',count(partCount(seriesDevices)))}</div><div class="section-heading"><h2>产品线</h2><span>第 2 层</span></div><div class="folder-list">${lines.map(([line,list])=>`<button class="folder-row" data-line="${esc(line)}"><span class="folder-icon">${esc(line.replace(browse.series,'').slice(0,3)||'MCU')}</span><span class="folder-main"><h3>${esc(line)}</h3><p>${[...new Set(list.map(d=>d.c||d.a).filter(Boolean))].join(' / ')||'核心未知'} · 最高 ${clock(maxClock(list))}</p></span><span class="folder-meta"><b>${count(list.length)}</b><span>变体 · ${partCount(list)} 订货号 <i class="folder-chevron">›</i></span></span></button>`).join('')}</div>`;
      bindCrumbs();document.querySelectorAll('[data-line]').forEach(b=>b.onclick=()=>{browse.line=b.dataset.line;$('#view').scrollTop=0;renderCatalog()});return;
    }

    const lineDevices=seriesDevices.filter(d=>d.l===browse.line).sort((a,b)=>natural(a.n,b.n));
    $('#view').innerHTML=`${breadcrumb([{level:'root',label:'芯片目录'},{level:'vendor',label:browse.vendor},{level:'series',label:browse.series},{level:'line',label:browse.line}])}<div class="page-heading"><h1>${esc(browse.line)}</h1><p>器件变体层保留厂商标注的封装、引脚、存储和版本后缀，不把不同变体合并成一个型号。</p></div><div class="summary-strip">${summary('器件变体',count(lineDevices.length))}${summary('完整订货号',count(partCount(lineDevices)))}${summary('最高主频',clock(maxClock(lineDevices)))}</div><div class="section-heading"><h2>具体器件变体</h2><span>第 3 层</span></div><div class="variant-list">${lineDevices.map(d=>`<button class="variant-row" data-device="${esc(d.id)}"><span class="variant-id"><h3>${missingInfoBadge(d)}${esc(d.n)}</h3><p>变体码 ${esc(d.v||'—')} · ${esc(d.c||d.a||'—')} · ${clock(d.hz)} · ${d.sercom!==undefined?'SERCOM/FLEXCOM '+value(d.sercom):'UART '+value(d.uart)} · ${memory(d.fl)} Flash · ${memory(d.ra)} RAM</p></span><span class="variant-side"><b>${value(d.idx)}</b><span>选型指数 ${(d.parts||[]).length?`<i class="variant-parts">${(d.parts||[]).length} 订货号</i>`:''}</span></span></button>`).join('')}</div>`;
    bindCrumbs();document.querySelectorAll('[data-device]').forEach(b=>b.onclick=()=>openDetail(b.dataset.device));
  }
  function bindCrumbs(){document.querySelectorAll('[data-crumb]').forEach(b=>b.onclick=()=>{const level=b.dataset.crumb;if(level==='root'){state.browse.vendor=null;state.browse.series=null;state.browse.line=null}else if(level==='vendor'){state.browse.series=null;state.browse.line=null}else if(level==='series'){state.browse.line=null}$('#view').scrollTop=0;renderCatalog()})}
  function filtered(){const q=state.query.trim().toLowerCase();const list=devices.filter(d=>(!q||d._q.includes(q))&&(!state.vendorFilter||d.m===state.vendorFilter)&&(!state.coreFilter||(d.c||d.a)===state.coreFilter)&&(!state.peripheralFilter||(peripheralCount(d,state.peripheralFilter)||0)>=state.peripheralMin));list.sort(state.sort==='name'?(a,b)=>natural(a.n,b.n):(a,b)=>(b.idx??-1)-(a.idx??-1)||natural(a.n,b.n));return list}
  function deviceRow(d){const selected=state.compare.has(d.id);const selectedPeripheral=peripheralByKey.get(state.peripheralFilter);const peripheralSpec=selectedPeripheral?`<span>${value(peripheralCount(d,selectedPeripheral.key))}<small>${esc(selectedPeripheral.label)}</small></span>`:`<span>${value(d.sercom!==undefined?d.sercom:(d.usart!==undefined?d.usart:d.uart))}<small>${d.sercom!==undefined?'SERCOM/FLEXCOM':d.usart!==undefined?'USART':'UART'}</small></span>`;return `<button class="device-row" data-device="${esc(d.id)}"><span><span class="device-title"><h3>${missingInfoBadge(d)}${esc(d.n)}</h3>${(d.parts||[]).length?`<i>${(d.parts||[]).length} 订货号</i>`:''}</span><p class="device-path">${esc(vendorName(d.m))} › ${esc(d.s)} › ${esc(d.l)}</p>${boardTags(d)}<span class="device-specs"><span>${esc(d.c||d.a||'—')}<small>核心</small></span><span>${clock(d.hz)}<small>主频</small></span><span>${memory(d.fl)}<small>Flash</small></span>${peripheralSpec}</span></span><span class="device-score"><b>${value(d.idx)}</b><span>选型指数</span><i class="compare-toggle ${selected?'selected':''}" data-compare="${esc(d.id)}">${selected?'已对比':'＋ 对比'}</i></span></button>`}
  function renderSearch(full=true){
    if(full)renderHeader();const list=filtered();
    const active=state.query||state.vendorFilter||state.coreFilter||state.peripheralFilter;
    $('#view').innerHTML=`<div class="page-heading"><h1>搜索与筛选</h1><p>支持型号、订货号、核心以及外设名称搜索，并可按外设数量筛选。</p></div><div class="filter-panel"><div class="select-wrap"><label>厂商</label><select id="vendor-filter"><option value="">全部厂商</option>${manufacturers.map(v=>`<option value="${esc(v)}" ${state.vendorFilter===v?'selected':''}>${esc(v)}</option>`).join('')}</select></div><div class="select-wrap"><label>核心</label><select id="core-filter"><option value="">全部核心</option>${cores.map(v=>`<option value="${esc(v)}" ${state.coreFilter===v?'selected':''}>${esc(v)}</option>`).join('')}</select></div><div class="select-wrap"><label>外设</label><select id="peripheral-filter"><option value="">全部外设</option>${peripheralFilters.map(item=>`<option value="${item.key}" ${state.peripheralFilter===item.key?'selected':''}>${esc(item.label)}</option>`).join('')}</select></div><div class="select-wrap"><label>最少数量</label><select id="peripheral-min" ${state.peripheralFilter?'':'disabled'}>${[1,2,3,4,8,16,32].map(v=>`<option value="${v}" ${state.peripheralMin===v?'selected':''}>≥ ${v}</option>`).join('')}</select></div></div><div class="filter-hint">关键词可直接输入 UART、CAN、USB OTG、摄像头、触摸、加密、蓝牙等外设名称。</div><div class="result-head"><b>${count(list.length)} 个匹配器件</b><span class="result-actions">${active?'<button id="clear-filters">清除条件</button>':''}<button id="sort-toggle">${state.sort==='score'?'按选型指数':'按型号'} ▾</button></span></div>${list.length?`<div class="device-list">${list.slice(0,state.limit).map(deviceRow).join('')}</div>${list.length>state.limit?`<button class="more-button" id="load-more">继续显示（剩余 ${count(list.length-state.limit)}）</button>`:''}`:'<div class="empty"><strong>没有找到匹配器件</strong>请降低外设数量，或清除部分筛选条件。</div>'}`;
    $('#vendor-filter').onchange=e=>{state.vendorFilter=e.target.value;state.limit=120;renderSearch(false)};$('#core-filter').onchange=e=>{state.coreFilter=e.target.value;state.limit=120;renderSearch(false)};$('#peripheral-filter').onchange=e=>{state.peripheralFilter=e.target.value;if(!state.peripheralFilter)state.peripheralMin=1;state.limit=120;renderSearch(false)};$('#peripheral-min').onchange=e=>{state.peripheralMin=Number(e.target.value)||1;state.limit=120;renderSearch(false)};if($('#clear-filters'))$('#clear-filters').onclick=()=>{state.query='';state.vendorFilter='';state.coreFilter='';state.peripheralFilter='';state.peripheralMin=1;state.limit=120;renderSearch(true)};$('#sort-toggle').onclick=()=>{state.sort=state.sort==='score'?'name':'score';renderSearch(false)};if($('#load-more'))$('#load-more').onclick=()=>{state.limit+=200;renderSearch(false)};bindDeviceRows();updateNav();
  }
  function bindDeviceRows(){document.querySelectorAll('#vendor-filter option').forEach(option=>{if(option.value)option.textContent=vendorName(option.value)});document.querySelectorAll('[data-device]').forEach(row=>row.onclick=()=>openDetail(row.dataset.device));document.querySelectorAll('[data-compare]').forEach(control=>control.onclick=e=>{e.stopPropagation();toggleCompare(control.dataset.compare);if(state.tab==='search')renderSearch(false)})}
  function toggleCompare(id){if(state.compare.has(id)){state.compare.delete(id);toast('已移出对比')}else{if(state.compare.size>=4){toast('最多同时对比 4 款');return}state.compare.add(id);toast('已加入对比')}saveCompare()}
  function modelTermHit(text,term){
    const needle=String(term||'').toLowerCase();
    if(!needle)return false;
    if(/[a-z0-9]/.test(needle)){
      const escaped=needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
      return new RegExp('(?:^|[^a-z0-9])'+escaped+'(?:[a-z0-9]*|$)','i').test(text);
    }
    return text.includes(needle);
  }
  function modelTextScore(text,terms){return (terms||[]).reduce((score,term)=>{const needle=String(term||'').toLowerCase();return score+(modelTermHit(text,needle)?Math.min(10,Math.max(2,needle.length)):0)},0)}
  function aiRelation(text,pattern){
    const source=String(text||''),chunks=source.split(/[;,。；！？!?]/),rx=new RegExp(pattern,'i');
    for(const chunk of chunks){
      const hit=rx.exec(chunk);if(!hit)continue;
      const before=chunk.slice(Math.max(0,hit.index-28),hit.index),after=chunk.slice(hit.index+hit[0].length,hit.index+hit[0].length+28);
      const negBefore=/(?:不要|别用|不想用|不是|不需要|无需|不含|排除|禁止|不考虑|不带|没有|不用|不想要|别|不配|不想配)[^，,;。]{0,14}$/i.test(before);
      const negAfter=/^[^，,;。]{0,5}(?:不要|别用|不想用|不是|不需要|无需|不含|排除|禁止|不考虑|不带|没有|不用|不想要|别|不配|不想配)/i.test(after);
      if(!negBefore&&!negAfter)continue;
      const soft=/(?:最好|尽量|尽可能|优先|倾向|建议|可以不|能不)[^，,;。]{0,18}$/i.test(before);
      const optional=/(?:也行|也可以|无所谓|没关系|可有可无|不强求|有就行|没有就行)/i.test(before+after);
      return {hard:!soft&&!optional,soft:soft&&!optional,optional};
    }
    return {hard:false,soft:false,optional:false};
  }
  function aiHasNegation(text,terms){return (terms||[]).some(term=>aiRelation(text,String(term||'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).hard)}
  function aiHasSoftNegation(text,terms){return (terms||[]).some(term=>aiRelation(text,String(term||'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).soft)}
  function aiHasOptional(text,pattern){
    if(aiRelation(text,pattern).optional)return true;
    const source=String(text||''),rx=new RegExp('(?:'+pattern+')[^，,;。]{0,10}(?:都行|都可以|随便|无所谓|有无均可|有没有都可以|可有可无)|(?:都行|都可以|随便|无所谓|有无均可|有没有都可以|可有可无)[^，,;。]{0,10}(?:'+pattern+')','i');
    return rx.test(source);
  }
  function aiHasSoftQualifier(text,pattern){
    const source=String(text||''),chunks=source.split(/[;,。；！？!?]/),rx=new RegExp(pattern,'i');
    return chunks.some(chunk=>{const hit=rx.exec(chunk);if(!hit)return false;const before=chunk.slice(Math.max(0,hit.index-24),hit.index),after=chunk.slice(hit.index+hit[0].length,hit.index+hit[0].length+16);return /(?:最好|尽量|尽可能|优先|倾向|希望|建议|能有|有了更好|有的话更好)/i.test(before+after)&&!/(?:不要|不需要|无需|不含|排除|禁止|不考虑|不带|没有|不用|不想要|别)/i.test(before)});
  }
  function aiNormalizeText(prompt){
    let text=String(prompt||'').toLowerCase().replace(/[，、；]/g,',').replace(/[。！？!?]/g,';').replace(/[\t\r\n]+/g,' ').replace(/\s+/g,' ').trim();
    const aliases=[
      [/^(?:帮我找个|帮我找一款|帮我找一个|我要一款|我要一个|我想要一款|有没有|给我推荐一个|给我推荐一款|推荐一个|推荐一款)\s*/g,' '],
      [/一两个|一两路?/g,' 1-2 '],
      [/两三个|两三路?/g,' 2-3 '],
      [/三四个|三四路?/g,' 3-4 '],
      [/四五个|四五路?/g,' 4-5 '],
      [/跑得快|跑得动|处理得快|反应快|响应要快|高频一点|频率高一点|性能别太差/g,' 高性能 '],
      [/性能高一些|性能越高越好|频率越高越好|主频越高越好|算力高一点/g,' 高性能 '],
      [/省电一点|吃电少|耗电少|功耗别太高|电池撑得久|续航长|续航久/g,' 低功耗 '],
      [/内存别太小|内存够用|ram大一点|内存大一点/g,' 大内存 '],
      [/接口多一点|外设多一点|多几个接口|接口丰富/g,' 外设丰富 '],
      [/资料多|资料全|文档多|好开发|容易开发|开发简单|好上手|上手快|生态好|工具链成熟/g,' 生态成熟 '],
      [/做电机控制板|做电机驱动/g,' 电机控制 '],
      [/做电池供电的传感器|电池供电传感器|便携传感器/g,' 低功耗 传感器 '],
      [/做一个带屏的小设备|带屏的小设备/g,' 显示界面 '],
      [/需要接摄像头|要接摄像头|接摄像头/g,' 摄像头 '],
      [/想做国产替代|做国产替代/g,' 国产替代 '],
      [/串行口|通信口|通讯口|调试口|调试串口|异步串口|多串口|串行接口/g,' uart '],
      [/两线接口|两线总线/g,' i2c '],
      [/串行外设接口/g,' spi '],
      [/通用串行总线|通用usb|usb接口/g,' usb '],
      [/低功耗蓝牙|蓝牙低功耗/g,' bluetooth '],
      [/无线网络|无线连接|无线局域网|无线上网/g,' wifi '],
      [/程序存储|程序空间|代码空间|闪存容量/g,' flash '],
      [/运行内存|片上内存|片上ram|内存容量/g,' ram '],
      [/主时钟|时钟频率|最高频率|运行频率/g,' 主频 '],
      [/浮点运算|硬浮点|浮点单元/g,' fpu '],
      [/计时器|定时资源|定时器资源|高级定时器/g,' timer '],
      [/定时器位宽|定时器宽度/g,' timer width '],
      [/采样速率|采样频率|采样速度/g,' sample rate '],
      [/双区闪存|双区flash|双bank(?:\s*flash)?|双 bank(?:\s*flash)?|dual[- ]bank(?:\s*flash)?/g,' dual-bank flash '],
      [/零等待闪存|零等待flash|zero[- ]wait(?:\s*flash)?/g,' zero-wait flash '],
      [/混合型闪存|杂合型闪存|hybrid(?:\s*flash)?/g,' hybrid flash '],
      [/百万次每秒|百万采样每秒|兆采样每秒/g,' msps '],
      [/千次每秒|千采样每秒/g,' ksps '],
      [/接显示屏|接屏|带显示|带屏|屏幕|显示屏|液晶|人机界面|图形界面/g,' display '],
      [/图像采集|图像传感器|视觉|摄像头接口/g,' camera ']
    ];
    aliases.forEach(([pattern,replacement])=>{text=text.replace(pattern,replacement)});
    return text.replace(/\s+/g,' ').trim();
  }
  function aiSingleNumberValue(raw){
    const token=String(raw??'').trim().toLowerCase();if(!token)return null;if(/^\d+(?:\.\d+)?$/.test(token))return Number(token);
    const digits={零:0,〇:0,一:1,二:2,两:2,三:3,四:4,五:5,六:6,七:7,八:8,九:9};if(token.length===1&&digits[token]!==undefined)return digits[token];
    let total=0,current=0;const units={十:10,百:100,千:1000,万:10000};for(const char of token){if(digits[char]!==undefined){current=digits[char];continue}if(units[char]){const unit=units[char];if(unit===10000){total+=(current||1)*unit;current=0}else{total+=(current||1)*unit;current=0}}else return null}return total+current||null;
  }
  function aiQuantityRange(raw){
    const token=String(raw??'').replace(/\s+/g,'').toLowerCase();if(!token)return null;
    const numeric=/^(\d+(?:\.\d+)?)[-~至到](\d+(?:\.\d+)?)$/.exec(token);
    if(numeric){const low=Number(numeric[1]),high=Number(numeric[2]);return Number.isFinite(low)&&Number.isFinite(high)?{min:Math.min(low,high),max:Math.max(low,high),target:(low+high)/2}:null}
    if(token.length===2&&/^[零〇一二两三四五六七八九][零〇一二两三四五六七八九]$/.test(token)){const low=aiSingleNumberValue(token[0]),high=aiSingleNumberValue(token[1]);return {min:Math.min(low,high),max:Math.max(low,high),target:(low+high)/2}}
    return null;
  }
  function aiNumberValue(raw){const range=aiQuantityRange(raw);return range?range.target:aiSingleNumberValue(raw)}
  function aiQuantityPattern(){return '((?:\\d+(?:\\.\\d+)?(?:\\s*[-~至到]\\s*\\d+(?:\\.\\d+)?)?|[零〇一二两三四五六七八九十百千万]+))'}
  function aiMaskBitWidthPeripheralText(source){
    let text=String(source||'');
    const number='(?:\\d+(?:\\.\\d+)?|[零〇一二两三四五六七八九十百千万]+)',bit='(?:位宽?|bits?)',peripheral='(?:adc|模数转换(?:器)?|dac|数模转换(?:器)?|timer|定时器|计数器)';
    // Bit width/resolution belongs to the peripheral, not its instance count.
    text=text.replace(new RegExp(number+'\\s*(?:[-‐‑–—]\\s*)?'+bit+'\\s*(?:的\\s*)?('+peripheral+')','ig'),(_match,kind)=>` ${kind} `);
    text=text.replace(new RegExp('('+peripheral+')[^，,;。]{0,12}?'+number+'\\s*(?:[-‐‑–—]\\s*)?'+bit,'ig'),(_match,kind)=>` ${kind} `);
    return text;
  }
  function aiNearConstraint(text,aliases){const number='(?:\\d+(?:\\.\\d+)?(?:\\s*[-~至到]\\s*\\d+(?:\\.\\d+)?)?|[零〇一二两三四五六七八九十百千万]+)',around='(?:约|大约|左右|上下|接近|差不多|附近)';return new RegExp(around+'[^，,;。]{0,16}(?:'+aliases+')|(?:'+aliases+')[^，,;。]{0,16}'+around+'|'+number+'\\s*(?:个|路|组|颗|项|通道)?\\s*'+around+'[^，,;。]{0,10}(?:'+aliases+')','i').test(text)}
  function aiModelInfer(text){
    const normalized=String(text||'').toLowerCase().replace(/[，、；]/g,',');
    const rank=(items)=>items.map(item=>({...item,score:modelTextScore(normalized,item.terms)})).filter(item=>item.score>0).sort((a,b)=>b.score-a.score);
    const vendorRank=rank(localModel.vendors||[]),coreRank=rank(localModel.cores||[]),profileRank=rank(localModel.profiles||[]),preferenceRank=rank(localModel.preferences||[]);
    const vendorClear=vendorRank[0]?.score>=3&&(!vendorRank[1]||vendorRank[0].score>=vendorRank[1].score+2);
    const coreClear=coreRank[0]?.score>=3&&(!coreRank[1]||coreRank[0].score>=coreRank[1].score+2);
    return {vendor:vendorClear?vendorRank[0].label:null,core:coreClear?coreRank[0].label:null,features:rank(localModel.features||[]).filter(item=>item.score>=3).map(item=>item.key),profiles:profileRank.filter(item=>item.score>=2),preferences:preferenceRank.filter(item=>item.score>=3),confidence:vendorRank[0]?.score||0};
  }
  function aiUnitBytes(number,unit){const n=aiNumberValue(number);if(!Number.isFinite(n))return null;const u=String(unit||'').toLowerCase();return n*(u==='gb'||u==='g'?1073741824:u==='mb'||u==='m'?1048576:u==='kb'||u==='k'?1024:1)}
  function aiFrequency(number,unit){const n=aiNumberValue(number);if(!Number.isFinite(n))return null;const u=String(unit||'mhz').toLowerCase();return n*(u==='ghz'||u==='g'?1e9:u==='mhz'||u==='m'?1e6:u==='khz'||u==='k'?1e3:1)}
  function aiConstraint(text,aliases){
    const a=String(aliases),source=String(text||''),number=aiQuantityPattern(),unit='(?:个|路|组|颗|项|通道|核)?',minWord='(?:至少|不少于|不低于|大于等于|起码|起步|最少|最低|需要|要有|得有|得至少|必须|必需|>=|≥|>)',maxWord='(?:最多|不超过|不高于|小于等于|至多|以下|以内|不能超过|不得超过|<=|≤|<)';
    const minBefore=new RegExp(minWord+'?\\s*'+number+'\\s*'+unit+'\\s*(?:以上|及以上)?\\s*(?:'+a+')','i').exec(source);
    const minAfter=new RegExp('(?:'+a+')\\s*'+minWord+'?\\s*'+number+'\\s*'+unit+'\\s*(?:以上|及以上)?','i').exec(source);
    const explicitMinBefore=new RegExp(minWord+'\\s*'+number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const explicitMinAfter=new RegExp('(?:'+a+')\\s*'+minWord+'\\s*'+number,'i').exec(source);
    const maxBefore=new RegExp(maxWord+'\\s*'+number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const maxAfter=new RegExp('(?:'+a+')[^，,;。]{0,12}'+maxWord+'\\s*'+number,'i').exec(source);
    const plainBefore=new RegExp(number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const plainAfter=new RegExp('(?:'+a+')\\s*'+number,'i').exec(source);
    const pick=(hit,mode)=>{if(!hit||hit[1]===undefined)return null;const range=aiQuantityRange(hit[1]);return range?(mode==='max'?range.max:mode==='target'?range.target:range.min):aiNumberValue(hit[1])};
    const explicitMin=pick(explicitMinBefore)||pick(explicitMinAfter),plain=plainBefore||plainAfter,plainRange=plain&&aiQuantityRange(plain[1]);
    const localMax=Boolean(maxBefore||maxAfter);
    const min=explicitMin||(!localMax?(pick(minBefore)||pick(minAfter)||pick(plainBefore)||pick(plainAfter)||null):null);
    const max=pick(maxBefore,'max')||pick(maxAfter,'max')||(!explicitMin&&!localMax&&plainRange?plainRange.max:null);
    const targetHit=new RegExp(number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source)||new RegExp('(?:'+a+')[^，,;。]{0,8}'+number+'\\s*'+unit,'i').exec(source);
    const targetHitRange=targetHit&&aiQuantityRange(targetHit[1]);
    const target=(targetHit&&(targetHitRange||aiNearConstraint(source,a)))?pick(targetHit,'target'):null;
    return {min,max,target,approx:Boolean(target)};
  }
  function aiMemory(text,aliases){const number=aiQuantityPattern(),unit='(gb|mb|kb|g|m|k|吉|兆)';const after=new RegExp('(?:'+aliases+')[^\\d零〇一二两三四五六七八九十百千万]{0,14}'+number+'\\s*'+unit,'i').exec(text);const before=new RegExp(number+'\\s*'+unit+'\\s*(?:'+aliases+')','i').exec(text);const m=after||before;return m?aiUnitBytes(m[1],m[2]==='吉'?'gb':m[2]==='兆'?'mb':m[2]):null}
  function aiCanonicalCore(token){
    const value=String(token||'').toLowerCase().replace(/[\s_-]+/g,'');
    const match=/m(?:35p|55|33|23|0\+?|7|4|3)/i.exec(value)||/m\d+\+?/i.exec(value);
    return match?'cortex-'+match[0].toLowerCase():null;
  }
  function aiCoreCandidates(text){
    const source=String(text||'').toLowerCase(),pattern=/(?:cortex[- ]?m(?:35p|55|33|23|0\+?|7|4|3)|arm\s+(?:cortex[- ]?)?m(?:35p|55|33|23|0\+?|7|4|3)|m(?:35p|55|33|23|0\+?|7|4|3)(?:f)?)(?=$|[^a-z0-9])/gi;
    const result=[],excluded=[],softExcluded=[];let hit;
    while((hit=pattern.exec(source))){
      const canonical=aiCanonicalCore(hit[0]);if(!canonical)continue;
      const escaped=hit[0].replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),relation=aiRelation(source,escaped);
      if(relation.hard){if(!excluded.includes(canonical))excluded.push(canonical)}
      else if(relation.soft){if(!softExcluded.includes(canonical))softExcluded.push(canonical)}
      else if(!result.includes(canonical))result.push(canonical);
    }
    return {cores:result,excluded,softExcluded};
  }
  function aiCoreFromText(text){
    const source=String(text||'').toLowerCase();
    // Longest-first plus an alphanumeric boundary prevents m33 from being cut to m3.
    const pattern=/(?:^|[^a-z0-9])((?:cortex[- ]?m(?:35p|55|33|23|0\+?|7|4|3)|arm\s+(?:cortex[- ]?)?m(?:35p|55|33|23|0\+?|7|4|3)|m(?:35p|55|33|23|0\+?|7|4|3)(?:f)?))(?=$|[^a-z0-9])/i;
    const hit=pattern.exec(source)||/(?:^|[^a-z0-9])((?:cortex[- ]?m|arm\s+(?:cortex[- ]?)?m|m)\d+\+?(?:f)?)(?=$|[^a-z0-9])/i.exec(source);if(!hit)return null;
    return aiCanonicalCore(hit[1]);
  }
  function aiRateValue(raw,unit){
    const n=aiNumberValue(raw);if(!Number.isFinite(n))return null;
    const u=String(unit||'').toLowerCase().replace(/\s+/g,'');
    if(/ghz/.test(u)||u==='g')return n*1e9;if(/mhz|msps|m\/s/.test(u)||u==='m')return n*1e6;if(/khz|ksps/.test(u)||u==='k')return n*1e3;return n;
  }
  const aiTechnicalEvidenceCache=new WeakMap();
  function aiEvidenceText(d){
    if(d&&aiTechnicalEvidenceCache.has(d))return aiTechnicalEvidenceCache.get(d);
    const inventory=(d?.pi||[]).flatMap(item=>[item?.n,item?.d,item?.t]).filter(Boolean);
    const memory=(d?.mem||[]).flatMap(item=>[item?.n,item?.s,item?.a]).filter(Boolean);
    const evidence=[inventory,memory,d?.acc,d?.feat,d?.pending].flat().filter(Boolean).join(' | ').toLowerCase();
    if(d)aiTechnicalEvidenceCache.set(d,evidence);return evidence;
  }
  function aiTechnicalMetric(d,key){
    if(!d)return null;
    if(key==='timerWidth'){
      const widths=[];const direct=Number(d.tw);if(Number.isFinite(direct)&&direct>0)widths.push(direct);
      const evidence=aiEvidenceText(d);
      for(const match of evidence.matchAll(/(?<!\d)(\d+)\s*-?\s*bit[^,;|]{0,20}(?:timer|定时器|计数器)|(?:timer|定时器|计数器)[^,;|]{0,20}(?<!\d)(\d+)\s*-?\s*bit/gi))widths.push(Number(match[1]||match[2]));
      return widths.length?Math.max(...widths):null;
    }
    if(key==='adcResolution'){
      const resolutions=[];const direct=Number(d.adr);if(Number.isFinite(direct)&&direct>0)resolutions.push(direct);
      for(const match of aiEvidenceText(d).matchAll(/(?<!\d)(\d+)\s*-?\s*bit[^,;|]{0,20}(?:adc|模数转换)|(?:adc|模数转换)[^,;|]{0,20}(?<!\d)(\d+)\s*-?\s*bit/gi))resolutions.push(Number(match[1]||match[2]));
      return resolutions.length?Math.max(...resolutions):null;
    }
    if(key==='dacResolution'){
      const match=/(\d+)\s*-?\s*bit[^,;|]{0,20}(?:dac|数模转换)|(?:dac|数模转换)[^,;|]{0,20}(\d+)\s*-?\s*bit/i.exec(aiEvidenceText(d));
      return match?Number(match[1]||match[2]):null;
    }
    if(key==='adcSampleRate'||key==='dacSampleRate'||key==='ioSpeed'){
      const directKey=key==='adcSampleRate'?'adcr':key==='dacSampleRate'?'dacr':'iospeed';
      if(d?.[directKey]!==undefined&&d?.[directKey]!==null&&d?.[directKey]!==''){
        const direct=Number(d[directKey]);if(Number.isFinite(direct)&&direct>0)return direct;
      }
      const source=aiEvidenceText(d),needle=key==='adcSampleRate'?'(?:adc|模数转换|采样|sample\\s*rate)':key==='dacSampleRate'?'(?:dac|数模转换)':'(?:gpio|i/o|io|翻转速度|引脚速度)';
      const rate='(\\d+(?:\\.\\d+)?)\\s*(ghz|mhz|khz|msps|ksps|m\\s*/\\s*s|sps|m|k)';
      const match=new RegExp(needle+'[^,;|]{0,35}?'+rate+'|'+rate+'[^,;|]{0,35}?'+needle,'i').exec(source);
      if(!match)return null;
      const number=match[1]||match[3],unit=match[2]||match[4];return aiRateValue(number,unit);
    }
    if(key==='flashWaitStates'){
      if(d?.fw!==undefined&&d?.fw!==null&&d?.fw!==''&&Number.isFinite(Number(d.fw)))return Number(d.fw);
      const source=aiEvidenceText(d);if(/zero[- ]?wait|0[- ]?wait|零等待|无等待/i.test(source))return 0;
      const match=/(\d+)\s*(?:[- ]?wait(?:ing)?\s*states?|等待(?:周期|状态)?)/i.exec(source);return match?Number(match[1]):null;
    }
    if(key==='flashBanks'){
      if(d?.fb!==undefined&&d?.fb!==null&&d?.fb!==''&&Number.isFinite(Number(d.fb)))return Number(d.fb);
      const source=aiEvidenceText(d);if(/dual[- ]?(?:bank|banked)|双\s*bank|双区|两区|两个?\s*flash\s*bank/i.test(source))return 2;if(/single[- ]?(?:bank|banked)|单\s*bank|单区/i.test(source))return 1;return null;
    }
    return null;
  }
  function aiTechnicalHas(d,key,term){
    const source=aiEvidenceText(d),needle=String(term||'').toLowerCase();
    if(!source)return null;
    if(key==='ramType'){
      const architecture=String(d?.ramarch||'');
      if(architecture)return new RegExp('(?:^|;)'+needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'(?:;|$)','i').test(architecture);
      if(!/(?:ram|sram|memory|itcm|dtcm|ccm|tcm|axi)/i.test(source))return null;
      if(!Array.isArray(d.mem)&&!new RegExp('(?:^|[^a-z0-9])'+needle+'(?:$|[^a-z0-9])','i').test(source))return null;
      return new RegExp('(?:^|[^a-z0-9])'+needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'(?:$|[^a-z0-9])','i').test(source);
    }
    if(key==='ramStructure'){
      if(String(d?.ramarch||''))return true;
      if(!/(?:ram|sram|memory|itcm|dtcm|ccm|tcm)/i.test(source))return null;
      if(Array.isArray(d.mem))return d.mem.length>1;
      return /itcm|dtcm|ccm|axi/i.test(source)?true:null;
    }
    if(key==='flashArchitecture'){
      if(needle==='dualbank'&&Number(d?.fb)===2)return true;
      if(needle==='singlebank'&&Number(d?.fb)===1)return true;
      if(needle==='zerowait'&&d?.fw!==undefined&&d?.fw!==null&&d?.fw!==''&&Number(d.fw)===0)return true;
      if(!/(?:flash|irom|bank|等待|wait)/i.test(source))return null;
      const aliases={dualBank:'dual[- ]?(?:bank|banked)|双\\s*bank|双区|两区',hybrid:'hybrid|混合型|杂合型',zeroWait:'zero[- ]?wait|零等待'};
      if(aliases[needle]){
        if(!new RegExp(Object.values(aliases).join('|'),'i').test(source))return null;
        return new RegExp(aliases[needle],'i').test(source);
      }
      return source.includes(needle);
    }
    if(key==='ramExclusive'){if(!/(?:ram|sram|memory|itcm|dtcm|ccm|tcm)/i.test(source))return null;return /exclusive|private|dedicated|独占|专用|私有|per-core|每核/i.test(source)?true:null}
    return source.includes(needle);
  }
  function aiParseTechnical(text,req){
    const source=String(text||''),number='(?:\\d+(?:\\.\\d+)?|[零〇一二两三四五六七八九十百千万]+)',bit='('+number+')\\s*(?:[-‐‑–—]\\s*)?(?:位宽?|bit(?:s)?)';
    const addRequirement=(label)=>{if(label&&!req.technicalRequirements.includes(label))req.technicalRequirements.push(label)};
    const timerWidth= new RegExp(bit+'\\s*(?:的\\s*)?(?:timer|定时器|计数器)|(?:timer|定时器|计数器)[^，,;。]{0,18}?'+bit,'i').exec(source);
    if(timerWidth){req.timerWidthMin=aiNumberValue(timerWidth[1]||timerWidth[2]);if(Number.isFinite(req.timerWidthMin)){req.minimums.tim=Math.max(req.minimums.tim||0,1);addRequirement(req.timerWidthMin+' 位定时器')}}
    const adcResolution=new RegExp(bit+'\\s*(?:的\\s*)?(?:adc|模数转换(?:器)?)|(?:adc|模数转换(?:器)?)[^，,;。]{0,18}?'+bit,'i').exec(source);
    if(adcResolution){req.adcResolution=aiNumberValue(adcResolution[1]||adcResolution[2]);if(Number.isFinite(req.adcResolution)){req.minimums.adch=Math.max(req.minimums.adch||0,1);addRequirement(req.adcResolution+' 位 ADC')}}
    const dacResolution=new RegExp(bit+'\\s*(?:的\\s*)?(?:dac|数模转换(?:器)?)|(?:dac|数模转换(?:器)?)[^，,;。]{0,18}?'+bit,'i').exec(source);
    if(dacResolution){req.dacResolution=aiNumberValue(dacResolution[1]||dacResolution[2]);if(Number.isFinite(req.dacResolution)){req.minimums.dac=Math.max(req.minimums.dac||0,1);addRequirement(req.dacResolution+' 位 DAC')}}
    const rate='(\\d+(?:\\.\\d+)?)\\s*(ghz|mhz|khz|msps|ksps|m\\s*/\\s*s|sps|m|k)';
    const rateFor=(needle)=>{const match=new RegExp('(?:'+needle+')[^，,;。]{0,28}?'+rate+'|'+rate+'[^，,;。]{0,28}?(?:'+needle+')','i').exec(source);if(!match)return null;return aiRateValue(match[1]||match[3],match[2]||match[4])};
    const adcRate=rateFor('adc|模数转换|采样(?:率|速度|速率)|sample\\s*rate');if(adcRate){req.adcSampleRate=adcRate;addRequirement('ADC 采样率 ≥ '+clock(adcRate))}
    const dacRate=rateFor('dac|数模转换');if(dacRate){req.dacSampleRate=dacRate;addRequirement('DAC 速度 ≥ '+clock(dacRate))}
    const ioRate=rateFor('gpio|i/o|io|IO翻转|引脚翻转|IO速度');if(ioRate){req.ioSpeed=ioRate;addRequirement('IO 速度 ≥ '+clock(ioRate))}
    const wait=/(?:flash|闪存)[^，,;。]{0,18}?(zero[- ]?wait|0[- ]?wait|零等待|(\\d+)\\s*(?:[- ]?wait(?:ing)?\\s*states?|等待(?:周期|状态)?))|(?:zero[- ]?wait|0[- ]?wait|零等待|(\\d+)\\s*(?:[- ]?wait(?:ing)?\\s*states?|等待(?:周期|状态)?))[^，,;。]{0,18}?(?:flash|闪存)/i.exec(source);
    const waitPhrase=/(zero[- ]?wait|0[- ]?wait|零等待|(\\d+)\\s*(?:[- ]?wait(?:ing)?\\s*states?|等待(?:周期|状态)?))/i.exec(source);
    const waitMatch=wait||(/(?:flash|闪存)/i.test(source)?waitPhrase:null);
    if(waitMatch){const value=/zero|0|零/i.test(waitMatch[1]||waitMatch[3]||'')?0:Number(waitMatch[2]||waitMatch[4]);if(Number.isFinite(value)){req.flashWaitStates=value;addRequirement('Flash '+(value===0?'零等待':'等待周期 ≤ '+value))}}
    if(/(?:双区|双 bank|dual[- ]?bank|dual[- ]?banked|两区)/i.test(source)){req.flashBanks=2;req.flashArchitecture.push('dualBank');addRequirement('双 Bank Flash')}
    if(/(?:单区|单 bank|single[- ]?bank)/i.test(source)){req.flashBanks=1;req.flashArchitecture.push('singleBank');addRequirement('单 Bank Flash')}
    if(/(?:混合型|杂合型|hybrid)\s*(?:flash|闪存)?/i.test(source)){req.flashArchitecture.push('hybrid');addRequirement('混合型 Flash')}
    const ramTerms=[['itcm','ITCM'],['dtcm','DTCM'],['ccm','CCM RAM'],['axi','AXI SRAM'],['sram1','SRAM1'],['sram2','SRAM2'],['tcm','TCM']];ramTerms.forEach(([term,label])=>{if(new RegExp('(?:^|[^a-z0-9])'+term+'(?:$|[^a-z0-9])','i').test(source)){req.ramTypes.push(term);addRequirement(label)}});
    const ramAlternative=new RegExp('(?:^|[^a-z0-9])(itcm|dtcm|ccm|axi|sram1|sram2|tcm)(?:$|[^a-z0-9])[^，,;。]{0,12}(?:或|or|/)[^，,;。]{0,12}(?:^|[^a-z0-9])(itcm|dtcm|ccm|axi|sram1|sram2|tcm)(?:$|[^a-z0-9])','i').exec(source);
    if(ramAlternative){req.ramTypeAny=[ramAlternative[1].toLowerCase(),ramAlternative[2].toLowerCase()];req.ramTypes=req.ramTypes.filter(type=>!req.ramTypeAny.includes(type));addRequirement(req.ramTypeAny.map(type=>type.toUpperCase()).join(' / ')+'（二选一）')}
    if(/(?:独占|专用|私有|每个核心.*ram|每核.*ram|per[- ]?core.*ram|private.*ram)/i.test(source)){req.ramExclusive=true;addRequirement('多核独占 RAM')}
    if(/(?:ram|sram|内存)[^，,;。]{0,10}(?:结构|分区|布局|类型|种类)|(?:结构|分区|布局|类型|种类)[^，,;。]{0,10}(?:ram|sram|内存)/i.test(source)){req.ramStructure=true;addRequirement('RAM 结构 / 分区')}
    if(/(?:adc|模数转换|采样)[^，,;。]{0,12}(?:快|高速|高速度|速度高|速度|速率)|(?:快|高速|高速度)[^，,;。]{0,12}(?:adc|模数转换|采样)/i.test(source))req.technicalPreferences.push('fastAdc');
    if(/(?:dac|数模转换)[^，,;。]{0,12}(?:快|高速|高速度|速度高|速度|速率)|(?:快|高速|高速度)[^，,;。]{0,12}(?:dac|数模转换)/i.test(source))req.technicalPreferences.push('fastDac');
    if(/(?:gpio|io|i\/o|引脚)[^，,;。]{0,12}(?:快|高速|高速度|速度高|速度|速率)|(?:快|高速|高速度)[^，,;。]{0,12}(?:gpio|io|i\/o|引脚)/i.test(source))req.technicalPreferences.push('fastIo');
    req.flashArchitecture=[...new Set(req.flashArchitecture)];req.ramTypes=[...new Set(req.ramTypes)];req.ramTypeAny=[...new Set(req.ramTypeAny||[])];req.technicalPreferences=[...new Set(req.technicalPreferences)];
  }
  function aiParsePower(text,req){
    const source=String(text||''),unit='(na|纳安|ua|µa|μa|微安|ma|毫安|a|安|uw|µw|μw|微瓦|mw|毫瓦|w|瓦)',number='(\\d+(?:\\.\\d+)?)';
    const pattern=new RegExp('(?:典型功耗|功耗|运行电流|工作电流|活动电流|待机电流|睡眠电流|睡眠功耗|停止电流|待机功耗|active current|run current|sleep current|standby current|power consumption)[^，,;。]{0,20}?'+number+'\\s*'+unit+'|'+number+'\\s*'+unit+'[^，,;。]{0,20}?(?:典型功耗|功耗|运行电流|工作电流|活动电流|待机电流|睡眠电流|睡眠功耗|停止电流|待机功耗|active current|run current|sleep current|standby current|power consumption)','gi');
    const normalize=(value,rawUnit)=>{const u=String(rawUnit||'').toLowerCase().replace(/μ|µ/g,'u');const n=Number(value);if(!Number.isFinite(n))return null;if(/^(?:na|纳安)$/.test(u))return {value:n/1000,basis:'current'};if(/^(?:ua|微安)$/.test(u))return {value:n,basis:'current'};if(/^(?:ma|毫安)$/.test(u))return {value:n*1000,basis:'current'};if(/^(?:a|安)$/.test(u))return {value:n*1e6,basis:'current'};if(/^(?:uw|微瓦)$/.test(u))return {value:n,basis:'power'};if(/^(?:mw|毫瓦)$/.test(u))return {value:n*1000,basis:'power'};if(/^(?:w|瓦)$/.test(u))return {value:n*1e6,basis:'power'};return null};
    let match;while((match=pattern.exec(source))){const value=match[1]||match[3],rawUnit=match[2]||match[4],normalized=normalize(value,rawUnit);if(!normalized)continue;const index=match.index,context=source.slice(Math.max(0,index-28),Math.min(source.length,index+match[0].length+28));const mode=/睡眠|待机|停止|休眠|sleep|standby/i.test(context)?'sleep':'run';const limit=/低于|小于|不超过|不高于|最多|以内|≤|<|under|below|less than/i.test(context);if(limit){req[mode==='sleep'?'powerSleepMax':'powerRunMax']={value:normalized.value,rawValue:Number(value),basis:normalized.basis,unit:String(rawUnit||'')};if(!req.technicalRequirements.includes((mode==='sleep'?'睡眠':'运行')+' '+value+' '+rawUnit+' 上限'))req.technicalRequirements.push((mode==='sleep'?'睡眠':'运行')+' '+value+' '+rawUnit+' 上限')}else{req.preferences=req.preferences||[];if(!req.preferences.includes('lowPower'))req.preferences.push('lowPower')}if(/典型功耗|典型电流|typical/i.test(context))req.powerTypicalOnly=true}
    if(/典型功耗|典型(?:运行|活动|待机|睡眠|停止)?电流|运行电流|待机电流|睡眠电流|power consumption|run current|sleep current|standby current/i.test(source)){req.powerMentioned=true;if(/典型(?:运行|活动|待机|睡眠|停止)?(?:功耗|电流)|typical/i.test(source))req.powerTypicalOnly=true;if(!req.preferences.includes('lowPower'))req.preferences.push('lowPower')}
  }
  function aiMetric(d,key){if(key==='serial'){const values=[d.uart,d.usart].filter(v=>typeof v==='number');return values.length?values.reduce((a,b)=>a+b,0):null}if(key==='usbAny'){const values=[d.usb,d.usbd,d.usbh,d.otg].filter(v=>typeof v==='number');return values.length?Math.max(...values):null}return peripheralCount(d,key)}
  function aiExactMatches(d,req){
    const query=String(req?.exact||'').toLowerCase(),haystack=String(d?._q||'').toLowerCase();
    if(!query)return true;
    if(haystack.includes(query))return true;
    const prefix=query.replace(/x+$/i,'');
    return prefix.length>=4&&haystack.includes(prefix);
  }
  function aiKnownWirelessAbsence(d,key){
    if(key!=='wifi'&&key!=='bluetooth')return false;
    // 对通用、低功耗和高性能 MCU，目录没有无线条目即表示未集成该无线能力；
    // 无线 MCU / SoC / 模组则必须保留“未核验”，避免把 Bluetooth 或 Wi-Fi 猜成不存在。
    const ordinary=['general_purpose_mcu','low_power_mcu','high_performance_mcu','micropython_mcu','dsp_mcu'];
    return ordinary.includes(d.pt)&&Array.isArray(d.pi)&&Number(d.cov)>=90;
  }
  function aiParse(prompt){
    const raw=String(prompt||'').trim(),text=aiNormalizeText(raw);
    const req={prompt:raw,normalized:text,vendor:null,excludedVendors:[],core:null,coreAny:[],excludedCores:[],softExcludedCores:[],exact:null,clock:null,clockMax:null,clockTarget:null,ram:null,ramMax:null,ramTarget:null,flash:null,flashMax:null,flashTarget:null,pins:null,pinsMin:null,pinsMax:null,coreCount:null,coreOnly:false,fpu:false,fpuExcluded:false,micropython:false,timerWidthMin:null,adcResolution:null,adcSampleRate:null,dacResolution:null,dacSampleRate:null,ioSpeed:null,flashWaitStates:null,flashBanks:null,flashArchitecture:[],ramTypes:[],ramTypeAny:[],ramExclusive:false,ramStructure:false,powerRunMax:null,powerSleepMax:null,powerBasis:null,powerTypicalOnly:false,powerMentioned:false,technicalRequirements:[],technicalPreferences:[],minimums:{},maximums:{},softMinimums:{},softTargets:{},excludedFeatures:[],softExcludedFeatures:[],vagueFeatures:[],features:[],profiles:[],profileLabels:[],preferences:[],warnings:[]};
    const modelIntent=aiModelInfer(text),addUnique=(list,value)=>{if(value&&!list.includes(value))list.push(value)};
    modelIntent.profiles.forEach(profile=>{addUnique(req.profiles,profile.key);addUnique(req.profileLabels,profile.label);Object.entries(profile.softMinimums||{}).forEach(([key,min])=>{if(!(key in req.minimums))req.softMinimums[key]=Math.max(req.softMinimums[key]||0,min)});(profile.preferences||[]).forEach(key=>addUnique(req.preferences,key))});
    modelIntent.preferences.forEach(preference=>addUnique(req.preferences,preference.key));
    const vendors=[
      ['STMicroelectronics',/stm32|stmicroelectronics|意法|st芯片/],['Espressif',/esp32|esp8266|乐鑫|espressif/],['Qinheng',/ch32|沁恒|wch|qingke|青稞/],['HPMicro',/hpmicro|hpm|先楫/],['Microchip',/microchip|atmel|avr|samd|pic/],['STC',/stc|宏晶/],['GigaDevice',/兆易创新|gigadevice|gd32/],['MindMotion',/灵动微|mindmotion|mm32/],['Nuvoton',/新唐|nuvoton|numicro/],['Puya',/普冉|puya|py32/],['Geehy',/极海|geehy|apm32/],['Infineon',/英飞凌|infineon|psoc|xmc/],['Texas Instruments',/德州仪器|ti芯片|texas instruments|mspm|msp430|c2000|tms320|f28[a-z0-9-]*|ti\s*dsp|\bdsp\b/],['Renesas',/瑞萨|renesas|\bra[02468][a-z0-9-]*\b|\brx[0-9][a-z0-9-]*\b|rl78|rh850|synergy/],['Artery',/雅特力|artery(?:tek)?|at32(?:f|a|l|m|wb)?\b/],['Allwinner',/全志|allwinner|xradio|xr806/],['MicroPy MCU',/micropython|micropy|canmv|rp2040|rp2350|rp2354|k210|k230|k510|树莓派|kendryte|嘉楠/]
    ];
    localModel.vendors.forEach(item=>{if(aiHasNegation(text,item.terms))req.excludedVendors.push(item.label)});
    const vendor=vendors.find(item=>item[1].test(text));if(modelIntent.vendor&&!req.excludedVendors.includes(modelIntent.vendor))req.vendor=modelIntent.vendor;else if(vendor&&!req.excludedVendors.includes(vendor[0]))req.vendor=vendor[0];
    const exact=/\b(?:tms320[a-z0-9-]+|f28[a-z0-9-]+|stm32|esp32|esp8266|ch32|gd32|mm32|py32|apm32|at32(?:f|a|l|m|wb)?|rp2040|rp2350[a-z0-9]*|rp2354[a-z0-9]*|k210|k230d?|k510|hpm[0-9a-z]+|ra[02468][a-z0-9-]*|rx[0-9][a-z0-9-]*|r7[a-z0-9-]+|r5f[a-z0-9-]+|r9a[a-z0-9-]+)[a-z0-9-]*\b/i.exec(text);if(exact)req.exact=exact[0].toLowerCase();
    const dspIntent=/(?:\bc2000\b|\btms320\b|\bf28[a-z0-9-]*\b|\bti\s*dsp\b|\bdsp\b|\bc28x\b|实时控制器)/i.test(text);
    if(dspIntent&&!req.vendor&&!req.excludedVendors.includes('Texas Instruments'))req.vendor='Texas Instruments';
    const coreIntent=aiCoreCandidates(text),coreAliases=[...cores].sort((a,b)=>b.length-a.length),core=coreAliases.find(item=>text.includes(String(item).toLowerCase())),explicitCore=aiCoreFromText(text);req.excludedCores.push(...coreIntent.excluded);req.softExcludedCores.push(...(coreIntent.softExcluded||[]));req.coreAny.push(...coreIntent.cores.filter(item=>!req.excludedCores.includes(item)&&!req.softExcludedCores.includes(item)));if(req.coreAny.length){req.core=req.coreAny[0]}else if(explicitCore&&!req.excludedCores.includes(explicitCore)&&!req.softExcludedCores.includes(explicitCore)){req.core=explicitCore;req.coreAny=[explicitCore]}else if(modelIntent.core&&!req.excludedCores.includes(modelIntent.core)&&!req.softExcludedCores.includes(modelIntent.core)){req.core=modelIntent.core;req.coreAny=[modelIntent.core]}else if(core){req.core=core;req.coreAny=[core]}else if(dspIntent){const dspCore=cores.find(item=>/c28x|dsp/i.test(item));if(dspCore){req.core=dspCore;req.coreAny=[dspCore]}}
    const quantity=aiQuantityPattern(),frequencyUnit='(ghz|mhz|khz|兆|m(?!b)|g(?!b)|k(?!b))',clockPattern=new RegExp('(?:主频|频率|时钟|最高|clock|速度)[^\\d零〇一二两三四五六七八九十百千万]{0,14}'+quantity+'\\s*'+frequencyUnit,'i'),clockPlain=new RegExp('(?:主频|频率|时钟|最高|clock)\\s*[:：=<>≤≥]?\\s*'+quantity+'(?![点些])','i'),clockFallback=new RegExp(quantity+'\\s*'+frequencyUnit,'i');let clockMatch=clockPattern.exec(text)||clockPlain.exec(text)||clockFallback.exec(text);if(clockMatch){let clockIndex=text.indexOf(clockMatch[0]),clockContext=text.slice(Math.max(0,clockIndex-16),clockIndex+clockMatch[0].length+18);if(/(?:adc|dac|gpio|io|采样|转换)/i.test(clockContext)&&!/(?:主频|时钟|最高)/i.test(clockContext)){const preferred=clockPlain.exec(text);if(preferred){clockMatch=preferred;clockIndex=text.indexOf(clockMatch[0]);clockContext=text.slice(Math.max(0,clockIndex-16),clockIndex+clockMatch[0].length+18)}}const unit=clockMatch[2]==='兆'?'mhz':clockMatch[2]||'mhz',clockValue=aiFrequency(clockMatch[1],unit);if(clockValue&&!(/(?:adc|dac|gpio|io|采样|转换)/i.test(clockContext)&&!/(?:主频|时钟|最高)/i.test(clockContext))){if(/最多|不超过|不高于|小于等于|以内|以下|别太高|不用太高|不要太高/.test(clockContext))req.clockMax=clockValue;else if(/约|大约|左右|上下|接近|差不多|附近/.test(clockContext))req.clockTarget=clockValue;else req.clock=clockValue}}
    const explicitClock=new RegExp('(?:主频|主时钟|时钟频率|运行频率|最高频率)\\s*[:：=<>≤≥]?\\s*'+quantity+'\\s*'+frequencyUnit,'i').exec(text);if(explicitClock){const explicitValue=aiFrequency(explicitClock[1],explicitClock[2]==='兆'?'mhz':explicitClock[2]||'mhz'),explicitIndex=text.indexOf(explicitClock[0]),explicitContext=text.slice(Math.max(0,explicitIndex-12),explicitIndex+explicitClock[0].length+12);if(/不超过|不高于|小于等于|以内|以下|最多|≤|<|别太高|不用太高|不要太高/.test(explicitContext))req.clockMax=explicitValue;else if(/约|大约|左右|上下|接近|差不多|附近/.test(explicitContext))req.clockTarget=explicitValue;else req.clock=explicitValue}
    const performanceRelation=aiRelation(text,'高频|高速|高性能|主频');
    if(/(?:主频|频率|速度)[^，,;。]{0,12}(?:越高|越快|高一些|高速)|高频|高速|高性能/i.test(text)&&!performanceRelation.hard&&!performanceRelation.soft)addUnique(req.preferences,'highPerformance');
    const memoryConstraint=(aliases)=>{const number=aiQuantityPattern(),unit='(gb|mb|kb|g|m|k|吉|兆)',after=new RegExp('(?:'+aliases+')[^\\d零〇一二两三四五六七八九十百千万]{0,14}'+number+'\\s*'+unit,'i'),before=new RegExp(number+'\\s*'+unit+'\\s*(?:'+aliases+')','i'),match=after.exec(text)||before.exec(text);if(!match)return null;const value=aiUnitBytes(match[1],match[2]==='吉'?'gb':match[2]==='兆'?'mb':match[2]),index=text.indexOf(match[0]),context=text.slice(Math.max(0,index-16),index+match[0].length+18);return {value,context}};
    const ramConstraint=memoryConstraint('ram|sram|内存'),flashConstraint=memoryConstraint('flash|闪存');if(ramConstraint?.value){if(/最多|不超过|不高于|小于等于|以内|以下|别太大|不用太大/.test(ramConstraint.context))req.ramMax=ramConstraint.value;else if(aiNearConstraint(text,'ram|sram|内存'))req.ramTarget=ramConstraint.value;else req.ram=ramConstraint.value}if(flashConstraint?.value){if(/最多|不超过|不高于|小于等于|以内|以下|别太大|不用太大/.test(flashConstraint.context))req.flashMax=flashConstraint.value;else if(aiNearConstraint(text,'flash|闪存'))req.flashTarget=flashConstraint.value;else req.flash=flashConstraint.value}
    const coreCountMatch=new RegExp('(单|双|三|四|五|六|七|八|\\d+)\\s*(?:核|核心)','i').exec(text);if(coreCountMatch){const map={单:1,双:2,三:3,四:4,五:5,六:6,七:7,八:8};req.coreCount=map[coreCountMatch[1]]||Number(coreCountMatch[1]);req.coreOnly=/单核|单核心|纯\s*核/i.test(text)}
    const pinAfter=new RegExp(quantity+'\\s*(?:pin|脚|引脚)','i').exec(text),pinBefore=new RegExp('(?:引脚|pin|脚)[^\\d零〇一二两三四五六七八九十百千万]{0,8}'+quantity,'i').exec(text),pinMatch=pinAfter||pinBefore;if(pinMatch){const pinValue=aiNumberValue(pinMatch[1]||pinMatch[2]),pinContext=text.slice(Math.max(0,text.indexOf(pinMatch[0])-8),text.indexOf(pinMatch[0])+pinMatch[0].length+8);if(/以内|以下|不超过|最多|小于等于/.test(pinContext))req.pinsMax=pinValue;else if(/至少|不少于|不低于|以上|大于等于/.test(pinContext))req.pinsMin=pinValue;else req.pins=pinValue}
    const fpuMention=/\bfpu\b|浮点|硬件浮点/i.test(text),fpuRelation=aiRelation(text,'\\bfpu\\b|浮点|硬件浮点');req.fpu=fpuMention&&!fpuRelation.hard;req.fpuExcluded=fpuMention&&fpuRelation.hard;req.micropython=/micropython|micropy|canmv|micro\s*python/i.test(text);
    const peripheralText=aiMaskBitWidthPeripheralText(text);
    const peripheralAliases=[
      ['serial','\\buart\\b|\\busart\\b|串口|串行|通信口'],['spi','\\bspi\\b|串行外设'],['i2c','\\bi2c\\b|i²c|两线总线'],['i2s','\\bi2s\\b|i²s|音频接口'],['can','\\bcan(?:fd)?\\b|\\btwai\\b|can总线'],['usbh','usb\\s*host|usb主机|host usb'],['usbd','usb\\s*device|usb设备|device usb'],['usbAny','\\busb\\b|usb设备|usb主机|otg'],['eth','ethernet|以太网'],['wifi','wi[- ]?fi|wifi|无线局域网|无线'],['bluetooth','bluetooth|蓝牙|\\bble\\b'],['cam','camera|摄像头|相机|dvp|dcmi'],['display','display|lcd|显示'],['pwm','\\bpwm\\b|脉宽|电机'],['adch','adc(?!\\s*(?:转换器|单元|converter))|模拟通道|adc通道|模拟输入通道'],['adcu','adc\\s*(?:转换器|单元|converter)|模数转换器|adc单元'],['gpio','\\bgpio\\b|通用io'],['tim','timer|定时器|计数器']
    ];
    peripheralAliases.forEach(([key,aliases])=>{
      const mentioned=new RegExp(aliases,'i').test(peripheralText),relation=aiRelation(peripheralText,aliases),optional=aiHasOptional(peripheralText,aliases),softMention=aiHasSoftQualifier(peripheralText,aliases),constraint=aiConstraint(peripheralText,aliases);
      if(relation.hard||optional){delete req.minimums[key];delete req.maximums[key];delete req.softTargets[key];if(relation.hard)addUnique(req.excludedFeatures,key);if(optional)addUnique(req.vagueFeatures,key);return}
      if(relation.soft){delete req.minimums[key];delete req.maximums[key];delete req.softTargets[key];addUnique(req.softExcludedFeatures,key);return}
      if(softMention){delete req.minimums[key];delete req.maximums[key];if(constraint.target!==null)req.softTargets[key]=constraint.target;if(constraint.min!==null)req.softMinimums[key]=Math.max(req.softMinimums[key]||0,constraint.min);else if(mentioned)req.softMinimums[key]=Math.max(req.softMinimums[key]||0,1);return}
      if(constraint.min!==null)req.minimums[key]=constraint.min;if(constraint.max!==null)req.maximums[key]=constraint.max;if(constraint.target!==null)req.softTargets[key]=constraint.target;
      if(constraint.min===null&&constraint.max===null&&!constraint.target&&mentioned){if(/够用|有就行|有即可|不用太多|随便几路|几路|若干|一些/i.test(text))addUnique(req.vagueFeatures,key);else req.minimums[key]=1}
    });
    if(req.minimums.usbh||req.minimums.usbd||req.maximums.usbh||req.maximums.usbd){delete req.minimums.usbAny;delete req.maximums.usbAny}
    modelIntent.features.forEach(key=>{const modelFeature=localModel.features.find(item=>item.key===key),pattern=(modelFeature?.terms||[]).map(term=>String(term).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),relation=pattern?aiRelation(text,pattern):{hard:false,soft:false,optional:false};if(relation.hard){addUnique(req.excludedFeatures,key);delete req.minimums[key];delete req.maximums[key]}else if(relation.soft){addUnique(req.softExcludedFeatures,key);delete req.minimums[key];delete req.maximums[key]}else if(key==='usbAny'&&(req.minimums.usbh||req.minimums.usbd)){}else if(req.vagueFeatures.includes(key)){}else if(!(key in req.minimums)&&!(key in req.softMinimums))req.minimums[key]=1});
    // “ADC 转换器/单元” is an instance count; it must never become an ADC
    // channel count merely because the shorter token “ADC” is also present.
    const adcUnitConstraint=aiConstraint(peripheralText,'adc\\s*(?:转换器|单元|converter)|模数转换器|adc单元');
    if(/(?:adc\\s*(?:转换器|单元|converter)|模数转换器|adc单元)/i.test(peripheralText)){
      delete req.minimums.adch;delete req.maximums.adch;delete req.softTargets.adch;
      if(adcUnitConstraint.min!==null)req.minimums.adcu=adcUnitConstraint.min;
      if(adcUnitConstraint.max!==null)req.maximums.adcu=adcUnitConstraint.max;
      if(adcUnitConstraint.min===null&&adcUnitConstraint.max===null&&!req.minimums.adcu&&!req.maximums.adcu)req.minimums.adcu=1;
    }
    const wirelessRelation=aiRelation(text,'无线|wifi|wi[- ]?fi');
    if(wirelessRelation.hard){delete req.minimums.wifi;delete req.minimums.bluetooth;addUnique(req.excludedFeatures,'wifi');if(!/保留\s*(?:蓝牙|bluetooth)/i.test(text))addUnique(req.excludedFeatures,'bluetooth')}
    else if(wirelessRelation.soft){delete req.minimums.wifi;delete req.minimums.bluetooth;addUnique(req.softExcludedFeatures,'wifi');if(!/保留\s*(?:蓝牙|bluetooth)/i.test(text))addUnique(req.softExcludedFeatures,'bluetooth')}
    const bluetoothRelation=aiRelation(text,'蓝牙|bluetooth|\\bble\\b');
    if(bluetoothRelation.hard){delete req.minimums.bluetooth;addUnique(req.excludedFeatures,'bluetooth')}
    else if(bluetoothRelation.soft){delete req.minimums.bluetooth;addUnique(req.softExcludedFeatures,'bluetooth')}
    Object.entries(req.softMinimums).forEach(([key,min])=>{if(key in req.minimums||key in req.maximums)delete req.softMinimums[key]});
    if(/便宜|价格低|成本低|低成本|性价比|省钱/i.test(text))req.warnings.push('目录不含实时价格，成本仅按封装、资源和生态作近似排序');
    if(/(?:最好|优先|倾向|尽量|希望|优先考虑)/i.test(text)&&/国产|国产替代|国内厂商/i.test(text))addUnique(req.preferences,'domestic');
    if(/大内存|内存大|内存别太小|内存够用|大容量|存储大/i.test(text))addUnique(req.preferences,'largeMemory');
    if(/外设丰富|接口多|接口多一点|串口多|多路接口|外设多/i.test(text))addUnique(req.preferences,'morePeripherals');
    if(/不要太大封装|封装别太大|封装小一点|小封装|小尺寸/i.test(text))addUnique(req.preferences,'compact');
    if(/资料多|资料全|文档多|好开发|容易开发|开发简单|好上手|上手快|生态成熟|工具链成熟/i.test(text))addUnique(req.preferences,'ecosystem');
    aiParseTechnical(text,req);
    aiParsePower(text,req);
    if(req.minimums.wifi)req.features.push('Wi-Fi');if(req.minimums.bluetooth)req.features.push('Bluetooth');if(req.minimums.cam)req.features.push('摄像头接口');if(req.minimums.display)req.features.push('显示接口');if(req.minimums.can)req.features.push('CAN');if(req.minimums.usbAny)req.features.push('USB');
    return req;
  }
  let aiEvaluationCacheRequest=null,aiEvaluationCache=new Map();
  function aiPowerLimitCheck(d,request){
    if(!request)return null;
    const candidate=powerMetric(d,{mode:request.mode,basis:request.basis,typicalOnly:Boolean(request.typicalOnly)});
    if(!candidate)return null;
    return candidate.value<=request.value;
  }
  function aiEvaluate(d,req){
    if(aiEvaluationCacheRequest!==req){aiEvaluationCacheRequest=req;aiEvaluationCache=new Map()}
    const cached=aiEvaluationCache.get(d.id);if(cached)return cached;
    const failures=[],unknowns=[],matched=[];const coreText=String(d.c||d.a||'').toLowerCase();
    const check=(condition,label,unknown)=>{if(unknown)unknowns.push(label);else if(condition)matched.push(label);else failures.push(label)};
    req.excludedVendors.forEach(vendor=>{if(d.m===vendor)failures.push('排除厂商 '+vendor)});
    if(req.vendor)check(d.m===req.vendor,'厂商 '+vendorName(req.vendor),!d.m);
    if(req.excludedCores?.length)req.excludedCores.forEach(core=>{if(coreText.includes(String(core).toLowerCase()))failures.push('排除核心 '+core)});
    if(req.coreAny?.length)check(req.coreAny.some(core=>coreText.includes(String(core).toLowerCase())),'核心 '+req.coreAny.join(' / '),!coreText);
    else if(req.core)check(coreText.includes(String(req.core).toLowerCase()),'核心 '+req.core,!coreText);
    if(req.coreOnly)check(d.cc===1&&!/[+/,]/.test(coreText),'单核',typeof d.cc!=='number');
    if(req.coreCount)check(d.cc===req.coreCount,req.coreCount+' 核',typeof d.cc!=='number');
    if(req.micropython)check(d.pt==='micropython_mcu','MicroPython',!d.pt);
    if(req.exact)check(aiExactMatches(d,req),'型号 '+req.exact,!d._q);
    if(req.clock)check(d.hz>=req.clock,'主频 ≥ '+clock(req.clock),typeof d.hz!=='number');
    if(req.clockMax)check(d.hz<=req.clockMax,'主频 ≤ '+clock(req.clockMax),typeof d.hz!=='number');
    if(req.ram)check(d.ra>=req.ram,'RAM ≥ '+memory(req.ram),typeof d.ra!=='number');
    if(req.ramMax)check(d.ra<=req.ramMax,'RAM ≤ '+memory(req.ramMax),typeof d.ra!=='number');
    if(req.flash)check(d.fl>=req.flash,'Flash ≥ '+memory(req.flash),typeof d.fl!=='number');
    if(req.flashMax)check(d.fl<=req.flashMax,'Flash ≤ '+memory(req.flashMax),typeof d.fl!=='number');
    if(req.pins)check(Number(d.pin)===req.pins,req.pins+' 引脚',d.pin===''||d.pin===undefined);
    if(req.pinsMin)check(Number(d.pin)>=req.pinsMin,req.pinsMin+' 引脚以上',d.pin===''||d.pin===undefined);
    if(req.pinsMax)check(Number(d.pin)<=req.pinsMax,req.pinsMax+' 引脚以内',d.pin===''||d.pin===undefined);
    if(req.fpu)check(d.fpu==='yes','FPU',!d.fpu||d.fpu==='unknown');
    if(req.fpuExcluded)check(d.fpu!=='yes','无 FPU',!d.fpu||d.fpu==='unknown');
    const technicalCheck=(key,required,label,mode='min')=>{if(required===null||required===undefined)return;const got=aiTechnicalMetric(d,key);const unknown=got===null||got===undefined;const ok=!unknown&&(mode==='max'?got<=required:got>=required);check(ok,label+' '+(mode==='max'?'≤ ':'≥ ')+required,unknown)};
    technicalCheck('timerWidth',req.timerWidthMin,'定时器位宽');
    technicalCheck('adcResolution',req.adcResolution,'ADC 分辨率');
    technicalCheck('adcSampleRate',req.adcSampleRate,'ADC 采样率');
    technicalCheck('dacResolution',req.dacResolution,'DAC 分辨率');
    technicalCheck('dacSampleRate',req.dacSampleRate,'DAC 速度');
    technicalCheck('ioSpeed',req.ioSpeed,'IO 速度');
    technicalCheck('flashWaitStates',req.flashWaitStates,'Flash 等待周期','max');
    technicalCheck('flashBanks',req.flashBanks,'Flash Bank 数');
    (req.flashArchitecture||[]).forEach(type=>check(aiTechnicalHas(d,'flashArchitecture',type),'Flash '+(type==='dualBank'?'双 Bank':type==='singleBank'?'单 Bank':'混合型'),aiTechnicalHas(d,'flashArchitecture',type)===null));
    if(req.ramTypeAny?.length){const evidence=req.ramTypeAny.map(type=>aiTechnicalHas(d,'ramType',type));const any=evidence.some(value=>value===true),unknown=evidence.some(value=>value===null);check(any,'RAM '+req.ramTypeAny.map(type=>type.toUpperCase()).join(' / ')+' 任一',unknown)}
    (req.ramTypes||[]).forEach(type=>check(aiTechnicalHas(d,'ramType',type),'RAM '+type.toUpperCase(),aiTechnicalHas(d,'ramType',type)===null));
    if(req.ramExclusive){const dedicated=aiTechnicalHas(d,'ramExclusive');check(dedicated===true,'多核独占 RAM',dedicated===null)}
    if(req.ramStructure){const structured=aiTechnicalHas(d,'ramStructure');check(structured===true,'RAM 结构 / 分区',structured===null)}
    if(req.powerRunMax){const ok=aiPowerLimitCheck(d,{...req.powerRunMax,mode:'run',typicalOnly:req.powerTypicalOnly});check(ok===true,'运行功耗 ≤ '+(req.powerRunMax.unit||'')+' '+(req.powerRunMax.rawValue??req.powerRunMax.value),ok===null)}
    if(req.powerSleepMax){const ok=aiPowerLimitCheck(d,{...req.powerSleepMax,mode:'sleep',typicalOnly:req.powerTypicalOnly});check(ok===true,'睡眠功耗 ≤ '+(req.powerSleepMax.unit||'')+' '+(req.powerSleepMax.rawValue??req.powerSleepMax.value),ok===null)}
    Object.entries(req.minimums).forEach(([key,min])=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);check(typeof got==='number'&&got>=min,label+' ≥ '+min,typeof got!=='number')});
    Object.entries(req.maximums).forEach(([key,max])=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);check(typeof got==='number'&&got<=max,label+' ≤ '+max,typeof got!=='number')});
    req.excludedFeatures.forEach(key=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got>0)failures.push('排除 '+label);else if(got===null&&!aiKnownWirelessAbsence(d,key))unknowns.push('排除 '+label+' 未核验')});
    const result={strict:!failures.length&&!unknowns.length,failures,unknowns,matched};aiEvaluationCache.set(d.id,result);return result;
  }
  function aiMeets(d,req){
    return aiEvaluate(d,req).strict;
  }
  function aiHardScope(d,req){
    const coreText=String(d.c||d.a||'').toLowerCase();
    if(req.excludedVendors.includes(d.m))return false;
    if(req.vendor&&d.m!==req.vendor)return false;
    if(req.excludedCores?.some(core=>coreText.includes(String(core).toLowerCase())))return false;
    if(req.coreAny?.length&&!req.coreAny.some(core=>coreText.includes(String(core).toLowerCase())))return false;
    if(req.core&&!req.coreAny?.length&&!coreText.includes(String(req.core).toLowerCase()))return false;
    if(req.coreOnly&&!(d.cc===1&&!/[+/,]/.test(coreText)))return false;
    if(req.coreCount&&d.cc!==req.coreCount)return false;
    if(req.micropython&&d.pt!=='micropython_mcu')return false;
    if(req.exact&&!aiExactMatches(d,req))return false;
    return true;
  }
  function aiDiverseSelect(items,req){
    if(req.exact)return items.slice(0,5);
    const selected=[],vendorCounts=new Map(),lineCounts=new Map();
    const add=(item,allowSameLine)=>{const vendor=item.device.m,line=item.device.l||item.device.s||item.device.n,key=vendor+'|'+line;const used=vendorCounts.get(vendor)||0;if(used>=2||(!allowSameLine&&(lineCounts.get(key)||0)>=1))return false;selected.push(item);vendorCounts.set(vendor,used+1);lineCounts.set(key,(lineCounts.get(key)||0)+1);return selected.length>=5};
    items.forEach(item=>{if(selected.length<5)add(item,false)});
    items.forEach(item=>{if(selected.length<5&&!selected.includes(item))add(item,true)});
    return selected;
  }
  const aiTechnicalPeakCache=new Map();
  function aiTechnicalPeak(metric){
    if(aiTechnicalPeakCache.has(metric))return aiTechnicalPeakCache.get(metric);
    const values=devices.map(item=>aiTechnicalMetric(item,metric)).filter(value=>typeof value==='number');
    const peak=values.length?Math.max(...values):0;aiTechnicalPeakCache.set(metric,peak);return peak;
  }
  function aiApplyTechnicalSignals(d,req,score,reasons){
    const add=(delta,label)=>{score+=delta;if(label)reasons.push(label)};
    const minSignals=[
      ['timerWidthMin','timerWidth','定时器位宽'],['adcResolution','adcResolution','ADC 分辨率'],['adcSampleRate','adcSampleRate','ADC 采样率'],
      ['dacResolution','dacResolution','DAC 分辨率'],['dacSampleRate','dacSampleRate','DAC 速度'],['ioSpeed','ioSpeed','IO 速度'],['flashBanks','flashBanks','Flash Bank 数']
    ];
    minSignals.forEach(([requiredKey,metric,label])=>{const required=req[requiredKey];if(required===null||required===undefined)return;const got=aiTechnicalMetric(d,metric);if(typeof got!=='number'){add(-3,label+' 未核验');return}if(got>=required)add(12,label+' 达标');else add(-14,label+' 偏低')});
    if(req.flashWaitStates!==null&&req.flashWaitStates!==undefined){const got=aiTechnicalMetric(d,'flashWaitStates');if(typeof got!=='number')add(-3,'Flash 等待周期未核验');else if(got<=req.flashWaitStates)add(12,'Flash 等待周期达标');else add(-14,'Flash 等待周期偏高')}
    (req.flashArchitecture||[]).forEach(type=>{const got=aiTechnicalHas(d,'flashArchitecture',type);const label=type==='dualBank'?'双 Bank Flash':type==='singleBank'?'单 Bank Flash':'混合型 Flash';if(got===null)add(-3,label+' 未核验');else if(got)add(12,label+' 命中');else add(-14,label+' 不符')});
    if(req.ramTypeAny?.length){const evidence=req.ramTypeAny.map(type=>aiTechnicalHas(d,'ramType',type));if(evidence.some(value=>value===true))add(12,'RAM '+req.ramTypeAny.map(type=>type.toUpperCase()).join(' / ')+' 命中');else if(evidence.some(value=>value===null))add(-3,'RAM 类型未核验');else add(-14,'RAM 类型不符')}
    (req.ramTypes||[]).forEach(type=>{const got=aiTechnicalHas(d,'ramType',type);if(got===null)add(-3,'RAM '+type.toUpperCase()+' 未核验');else if(got)add(12,'RAM '+type.toUpperCase());else add(-14,'缺少 RAM '+type.toUpperCase())});
    if(req.ramExclusive){const got=aiTechnicalHas(d,'ramExclusive');if(got===null)add(-3,'独占 RAM 未核验');else if(got)add(12,'独占 RAM');else add(-14,'无独占 RAM 证据')}
    if(req.ramStructure){const got=aiTechnicalHas(d,'ramStructure');if(got===null)add(-3,'RAM 结构未核验');else if(got)add(8,'RAM 结构已知');else add(-8,'RAM 结构单一')}
    const powerSignal=(request,label)=>{if(!request)return;const candidate=powerMetric(d,{mode:label==='运行'?'run':'sleep',basis:request.basis,typicalOnly:Boolean(req.powerTypicalOnly)});if(!candidate){add(-3,label+'功耗未核验');return}if(candidate.value<=request.value)add(12,label+'功耗达标');else add(-14,label+'功耗偏高')};
    powerSignal(req.powerRunMax,'运行');powerSignal(req.powerSleepMax,'睡眠');
    const fastSignals=[['fastAdc','adcSampleRate','ADC 速度'],['fastDac','dacSampleRate','DAC 速度'],['fastIo','ioSpeed','IO 速度']];
    (req.technicalPreferences||[]).forEach(pref=>{const signal=fastSignals.find(item=>item[0]===pref);if(!signal)return;const got=aiTechnicalMetric(d,signal[1]),peak=aiTechnicalPeak(signal[1]);if(typeof got!=='number'){add(-1,signal[2]+'未核验');return}if(peak&&got>=peak*.75)add(8,signal[2]+'领先');else if(peak&&got>=peak*.4)add(3,signal[2]+'较快')});
    return {score,reasons};
  }
  function aiApplySoftSignals(d,req,score,reasons){
    const prefs=Array.isArray(req.preferences)?req.preferences:[],softMinimums=req.softMinimums||{},softTargets=req.softTargets||{},softExcluded=Array.isArray(req.softExcludedFeatures)?req.softExcludedFeatures:[],add=(delta,label)=>{score+=delta;if(label)reasons.push(label)};
    (req.softExcludedCores||[]).forEach(core=>{if(String(d.c||d.a||'').toLowerCase().includes(String(core).toLowerCase()))add(-8,'尽量不含 '+core)});
    if(prefs.includes('lowPower')){if(d.pt==='low_power_mcu')add(12,'低功耗系列');else if(d.pt==='general_purpose_mcu'&&d.hz&&d.hz<=80000000)add(4,'较低运行功耗倾向');else if(d.hz&&d.hz>200000000)add(-6,'频率较高')}
    if(prefs.includes('highPerformance')){if(d.pt==='high_performance_mcu')add(10,'高性能系列');if(d.hz)add(Math.min(16,Math.max(1,Math.round(d.hz/25000000))),'主频性能优先')}
    if(prefs.includes('compact')){const pins=Number(d.pin);if(pins&&pins<=32)add(10,'小封装倾向');else if(pins&&pins<=48)add(6,'封装尺寸倾向');else if(pins>100)add(-4,'引脚数偏多')}
    if(prefs.includes('ecosystem')){if((d.boards||[]).length)add(8,'开发板生态');else if((d.parts||[]).length)add(3,'订货信息完整')}
    if(prefs.includes('domestic')&&['Artery','HPMicro','Qinheng','GigaDevice','Geehy','MindMotion','Nuvoton','Puya','STC','Allwinner'].includes(d.m))add(10,'国产厂商优先')
    if(prefs.includes('largeMemory')){const capacity=(Number(d.ra)||0)+(Number(d.fl)||0);if(capacity>=1048576)add(8,'容量优先');else if(capacity>=524288)add(4,'容量尚可')}
    if(prefs.includes('morePeripherals')){const count=['uart','usart','spi','i2c','can','usb','tim','pwm','adch'].reduce((sum,key)=>sum+(typeof d[key]==='number'?d[key]:0),0);if(count>=12)add(8,'外设丰富');else if(count>=6)add(4,'外设较多')}
    if(prefs.includes('wireless')){const wifi=aiMetric(d,'wifi'),bluetooth=aiMetric(d,'bluetooth');if(typeof wifi==='number'||typeof bluetooth==='number')add(8,'无线能力');else if(String(d.pt||'').startsWith('wireless'))add(4,'无线产品类型')}
    softExcluded.forEach(key=>{const got=aiMetric(d,key),label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got>0)add(-8,'尽量不含 '+label)});
    Object.entries(softMinimums).forEach(([key,min])=>{const got=aiMetric(d,key),label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got>=min)add(6,label+' 场景匹配');else if(typeof got==='number')add(-4,label+' 场景偏弱')});
    Object.entries(softTargets).forEach(([key,target])=>{const got=aiMetric(d,key),label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got!=='number')return;const ratio=Math.abs(got-target)/Math.max(1,target);if(ratio<=0.25)add(8,label+' 接近目标');else if(ratio<=0.5)add(3,label+' 略偏目标');else add(-Math.min(12,Math.max(4,Math.round(ratio*4))),label+' 偏离目标')});
    const targets=[['clockTarget',d.hz,'主频'],['ramTarget',d.ra,'RAM'],['flashTarget',d.fl,'Flash']];targets.forEach(([key,got,label])=>{const target=req[key];if(!target||typeof got!=='number')return;const ratio=Math.abs(got-target)/Math.max(1,target);if(ratio<=0.25)add(8,label+' 接近目标');else if(ratio<=0.5)add(3,label+' 略偏目标');else add(-Math.min(12,Math.max(4,Math.round(ratio*4))),label+' 偏离目标')});
    return {score,reasons};
  }
  function aiRecommend(prompt){
    const req=aiParse(prompt),pool=devices.slice(),scoped=pool.filter(d=>aiHardScope(d,req));let direct=scoped.filter(d=>aiMeets(d,req));let relaxed=false,scopeUnavailable=!scoped.length;
    if(!direct.length&&!scopeUnavailable){relaxed=true;direct=scoped}
    if(scopeUnavailable){return {req,relaxed:false,scopeUnavailable:true,text:`当前目录没有找到可核验的 ${req.core||req.vendor||req.exact||'目标范围'} 器件，已停止跨核心 / 跨厂商推荐，避免给出看似相近但实际不符合的结果。请放宽核心、厂商或型号条件后重试。`,results:[]}}
    let scored=direct.map(d=>{let score=(Number(d.idx)||0)*.38+(Number(d.cov)||0)*.16+((d.parts||[]).length?4:0);const reasons=[];const deviceCore=String(d.c||d.a||'').toLowerCase();if(req.vendor){if(d.m===req.vendor){score+=18;reasons.push('厂商匹配')}else score-=14}if(req.coreAny?.length){if(req.coreAny.some(core=>deviceCore.includes(String(core).toLowerCase()))){score+=15;reasons.push('核心匹配')}else score-=12}else if(req.core){if(deviceCore.includes(String(req.core).toLowerCase())){score+=15;reasons.push('核心匹配')}else score-=12}if(req.micropython){if(d.pt==='micropython_mcu'){score+=18;reasons.push('MicroPython 生态')}else score-=20}if(req.exact&&aiExactMatches(d,req)){score+=35;reasons.push('型号命中')}if(req.clock){if(d.hz>=req.clock){score+=12;reasons.push(clock(d.hz)+' 达标')}else if(d.hz)score-=18;else reasons.push('主频未核验')}if(req.clockMax){if(d.hz<=req.clockMax){score+=8;reasons.push(clock(d.hz)+' 未超上限')}else if(d.hz)score-=14;else reasons.push('主频未核验')}if(req.ram){if(d.ra>=req.ram){score+=8;reasons.push(memory(d.ra)+' RAM')}else if(d.ra)score-=14;else reasons.push('RAM 未核验')}if(req.ramMax){if(d.ra<=req.ramMax){score+=6;reasons.push('RAM 在上限内')}else if(d.ra)score-=10;else reasons.push('RAM 未核验')}if(req.flash){if(d.fl>=req.flash){score+=6;reasons.push(memory(d.fl)+' Flash')}else if(d.fl)score-=10;else reasons.push('Flash 未核验')}if(req.flashMax){if(d.fl<=req.flashMax){score+=5;reasons.push('Flash 在上限内')}else if(d.fl)score-=9;else reasons.push('Flash 未核验')}if(req.fpu){if(d.fpu==='yes'){score+=10;reasons.push('FPU')}else if(d.fpu==='no')score-=18;else reasons.push('FPU 未核验')}if(req.fpuExcluded){if(d.fpu==='no'){score+=8;reasons.push('无 FPU')}else if(d.fpu==='yes')score-=12;else reasons.push('FPU 未核验')}if(req.pins){if(Number(d.pin)===req.pins){score+=5;reasons.push(req.pins+' 引脚')}else if(d.pin)score-=4}if(req.pinsMin){if(Number(d.pin)>=req.pinsMin){score+=5;reasons.push(req.pinsMin+' 引脚以上')}else if(d.pin)score-=4}if(req.pinsMax){if(Number(d.pin)<=req.pinsMax){score+=5;reasons.push(req.pinsMax+' 引脚以内')}else if(d.pin)score-=4}Object.entries(req.minimums).forEach(([key,min])=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got>=min){score+=10;reasons.push(label+' '+got+' 路')}else if(typeof got==='number')score-=12;else reasons.push(label+' 未核验')});Object.entries(req.maximums).forEach(([key,max])=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got<=max){score+=7;reasons.push(label+' 在上限内')}else if(typeof got==='number')score-=10;else reasons.push(label+' 未核验')});const technical=aiApplyTechnicalSignals(d,req,score,reasons);score=technical.score;reasons.splice(0,reasons.length,...technical.reasons);const soft=aiApplySoftSignals(d,req,score,reasons);score=soft.score;reasons.splice(0,reasons.length,...soft.reasons);if(!reasons.length)reasons.push('选型指数 '+value(d.idx));return {device:d,score:Math.max(0,Math.min(100,Math.round(score))),reasons:reasons.slice(0,5)}}).sort((a,b)=>b.score-a.score||((b.device.idx||0)-(a.device.idx||0))||natural(a.device.n,b.device.n));
    scored=scored.map(item=>{const evaluation=aiEvaluate(item.device,req);const fail=evaluation.failures.map(label=>'不满足 '+label),unknown=evaluation.unknowns.map(label=>'未核验 '+label);const penalty=fail.length*24+unknown.length*14;return {...item,strict:evaluation.strict,violations:fail,unknowns:unknown,score:Math.max(0,Math.min(100,item.score-penalty)),reasons:[...fail,...unknown,...item.reasons].slice(0,5)}}).sort((a,b)=>b.score-a.score||((b.device.idx||0)-(a.device.idx||0))||natural(a.device.n,b.device.n));
    const selected=aiDiverseSelect(scored,req);
    const known=Object.keys(req.minimums).length+Object.keys(req.maximums).length+Object.keys(req.softMinimums||{}).length+Object.keys(req.softTargets||{}).length+Number(Boolean(req.vendor))+Number(Boolean(req.coreAny?.length||req.core))+Number(Boolean(req.excludedCores?.length))+Number(Boolean(req.softExcludedCores?.length))+Number(Boolean(req.coreCount))+Number(Boolean(req.clock))+Number(Boolean(req.clockMax))+Number(Boolean(req.clockTarget))+Number(Boolean(req.ram))+Number(Boolean(req.ramMax))+Number(Boolean(req.ramTarget))+Number(Boolean(req.flash))+Number(Boolean(req.flashMax))+Number(Boolean(req.flashTarget))+Number(Boolean(req.pins))+Number(Boolean(req.pinsMin))+Number(Boolean(req.pinsMax))+Number(Boolean(req.fpu))+Number(Boolean(req.fpuExcluded))+Number(Boolean(req.micropython))+Number(Boolean(req.timerWidthMin))+Number(Boolean(req.adcResolution))+Number(Boolean(req.adcSampleRate))+Number(Boolean(req.dacResolution))+Number(Boolean(req.dacSampleRate))+Number(Boolean(req.ioSpeed))+Number(req.flashWaitStates!==null&&req.flashWaitStates!==undefined)+Number(Boolean(req.flashBanks))+Number(Boolean(req.flashArchitecture?.length))+Number(Boolean(req.ramTypes?.length))+Number(Boolean(req.ramTypeAny?.length))+Number(Boolean(req.ramExclusive))+Number(Boolean(req.ramStructure))+Number(Boolean(req.technicalPreferences?.length))+req.profiles.length+req.preferences.length;const scope=known?'按自然语言理解出的场景、偏好与硬约束排序':'按本地模型与数据完整度排序';const warning=req.warnings.length?' '+req.warnings.join(' '):'';const text=relaxed?'没有找到全部满足且字段已核验的器件，以下仅列出同一核心 / 厂商范围内违约项最少的近似候选；请优先查看“未核验 / 不满足”提示。'+warning:`${scope}，给出 ${selected.length} 款候选。${warning}`;return {req,relaxed,text,results:selected};
  }
  function aiContextualPrompt(prompt){
    const current=String(prompt||'').trim();if(!current)return current;
    const previous=[...state.assistantMessages].reverse().find(message=>message&&message.role==='assistant'&&message.req&&message.req.prompt);
    if(!previous)return current;
    // Only carry context for additive follow-ups; a fresh request or an explicit replacement starts clean.
    if(/^(?:还要|还需要|另外|再加|再要|同时|并且|以及|不要|别要|排除|去掉|保留|最好|优先|也要|也想要)/i.test(current))return `${previous.req.prompt}，${current}`;
    return current;
  }
  function aiConstraintText(req){
    // 历史记录可能来自旧版本，字段类型和当前请求对象不完全一致；展示层必须容错。
    const source=req&&typeof req==='object'?req:{};
    const parts=[];
    const excludedVendors=Array.isArray(source.excludedVendors)?source.excludedVendors:[];
    const minimums=source.minimums&&typeof source.minimums==='object'?source.minimums:{};
    const maximums=source.maximums&&typeof source.maximums==='object'?source.maximums:{};
    const excludedFeatures=Array.isArray(source.excludedFeatures)?source.excludedFeatures:[];
    const softExcludedFeatures=Array.isArray(source.softExcludedFeatures)?source.softExcludedFeatures:[];
    const softMinimums=source.softMinimums&&typeof source.softMinimums==='object'?source.softMinimums:{};
    const softTargets=source.softTargets&&typeof source.softTargets==='object'?source.softTargets:{};
    const profileLabels=Array.isArray(source.profileLabels)?source.profileLabels:[];
    const preferences=Array.isArray(source.preferences)?source.preferences:[];
    if(source.vendor)parts.push(vendorName(source.vendor));
    if(excludedVendors.length)parts.push('排除 '+excludedVendors.map(vendorName).join('、'));
    if(source.coreAny?.length)parts.push('核心 '+source.coreAny.join(' 或 '));else if(source.core)parts.push(source.core);
    if(source.excludedCores?.length)parts.push('排除核心 '+source.excludedCores.join('、'));
    if(source.softExcludedCores?.length)parts.push('尽量不含核心 '+source.softExcludedCores.join('、'));
    if(source.coreCount)parts.push(source.coreCount+' 核');
    if(source.coreOnly)parts.push('单核');
    if(source.clock)parts.push('主频 ≥ '+clock(source.clock));
    if(source.clockMax)parts.push('主频 ≤ '+clock(source.clockMax));
    if(source.ram)parts.push('RAM ≥ '+memory(source.ram));
    if(source.ramMax)parts.push('RAM ≤ '+memory(source.ramMax));
    if(source.flash)parts.push('Flash ≥ '+memory(source.flash));
    if(source.flashMax)parts.push('Flash ≤ '+memory(source.flashMax));
    if(source.pins)parts.push(source.pins+' 引脚');
    if(source.pinsMin)parts.push('引脚 ≥ '+source.pinsMin);
    if(source.pinsMax)parts.push('引脚 ≤ '+source.pinsMax);
    if(source.fpu)parts.push('FPU');
    if(source.fpuExcluded)parts.push('不含 FPU');
    if(source.micropython)parts.push('MicroPython');
    if(source.clockTarget)parts.push('主频约 '+clock(source.clockTarget));
    if(source.ramTarget)parts.push('RAM约 '+memory(source.ramTarget));
    if(source.flashTarget)parts.push('Flash约 '+memory(source.flashTarget));
    if(source.timerWidthMin)parts.push('定时器位宽 ≥ '+source.timerWidthMin+' bit');
    if(source.adcResolution)parts.push('ADC ≥ '+source.adcResolution+' bit');
    if(source.adcSampleRate)parts.push('ADC 采样率 ≥ '+clock(source.adcSampleRate));
    if(source.dacResolution)parts.push('DAC ≥ '+source.dacResolution+' bit');
    if(source.dacSampleRate)parts.push('DAC 速度 ≥ '+clock(source.dacSampleRate));
    if(source.ioSpeed)parts.push('IO 速度 ≥ '+clock(source.ioSpeed));
    if(source.flashWaitStates!==null&&source.flashWaitStates!==undefined)parts.push('Flash 等待周期 ≤ '+source.flashWaitStates);
    if(source.flashBanks)parts.push('Flash '+source.flashBanks+' Bank');
    if(source.flashArchitecture?.length)parts.push('Flash '+source.flashArchitecture.join(' / '));
    if(source.ramTypes?.length)parts.push('RAM '+source.ramTypes.map(type=>type.toUpperCase()).join(' / '));
    if(source.ramTypeAny?.length)parts.push('RAM '+source.ramTypeAny.map(type=>type.toUpperCase()).join(' / ')+' 任一');
    if(source.ramExclusive)parts.push('RAM 需多核独占区');
    if(source.ramStructure)parts.push('RAM 结构 / 分区');
    if(source.powerRunMax)parts.push('运行功耗 ≤ '+(source.powerRunMax.rawValue??source.powerRunMax.value)+' '+(source.powerRunMax.unit||''));
    if(source.powerSleepMax)parts.push('睡眠功耗 ≤ '+(source.powerSleepMax.rawValue??source.powerSleepMax.value)+' '+(source.powerSleepMax.unit||''));
    if(source.powerTypicalOnly)parts.push('仅看典型功耗');
    const technicalPreferenceLabels={fastAdc:'ADC 高速',fastDac:'DAC 高速',fastIo:'IO 高速'};
    if(source.technicalPreferences?.length)parts.push('技术偏好 '+source.technicalPreferences.map(key=>technicalPreferenceLabels[key]||key).join('、'));
    Object.entries(minimums).forEach(([key,n])=>parts.push((key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key)+' ≥ '+n));
    Object.entries(maximums).forEach(([key,n])=>parts.push((key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key)+' ≤ '+n));
    Object.entries(softMinimums).forEach(([key,n])=>parts.push('场景需要 '+(key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key)+' '+n+' 路'));
    Object.entries(softTargets).forEach(([key,n])=>parts.push((key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key)+' 约 '+n+' 路'));
    if(profileLabels.length)parts.push('场景 '+profileLabels.join('、'));
    const preferenceLabels={lowPower:'低功耗',highPerformance:'高性能',compact:'小封装',ecosystem:'成熟生态',domestic:'国产优先',wireless:'无线能力',largeMemory:'大容量',morePeripherals:'外设丰富'};
    if(preferences.length)parts.push('偏好 '+preferences.map(key=>preferenceLabels[key]||key).join('、'));
    if(excludedFeatures.length)parts.push('不含 '+excludedFeatures.map(key=>key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key).join('、'));
    if(softExcludedFeatures.length)parts.push('尽量不含 '+softExcludedFeatures.map(key=>key==='serial'?'串口':key==='usbAny'?'USB':peripheralByKey.get(key)?.label||key).join('、'));
    return parts.length?parts.join(' · '):'未设置硬约束';
  }
  function aiResultCard(item){const d=item.device;const selected=state.compare.has(d.id);return `<div class="assistant-result ${item.strict?'strict':'approximate'}" data-ai-device="${esc(d.id)}" role="button" tabindex="0"><div class="assistant-result-main"><div class="assistant-result-title"><b>${esc(d.n)}</b><span>${item.strict?'满足':'近似'} · ${item.score} / 100</span></div><p>${esc(vendorName(d.m))} › ${esc(d.s)} › ${esc(d.l)}</p><div class="assistant-result-specs"><span>${esc(d.c||d.a||'—')}</span><span>${clock(d.hz)}</span><span>${memory(d.fl)} Flash</span><span>${memory(d.ra)} RAM</span></div><div class="assistant-reasons">${item.reasons.map(reason=>`<i class="${reason.startsWith('不满足')||reason.startsWith('未核验')?'warning':''}">${esc(reason)}</i>`).join('')}</div></div><button class="assistant-compare ${selected?'selected':''}" data-ai-compare="${esc(d.id)}">${selected?'已对比':'＋ 对比'}</button></div>`}
  function aiMessageHtml(message){if(message.role==='user')return `<div class="assistant-message user"><div>${esc(message.text)}</div></div>`;return `<div class="assistant-message bot"><div class="assistant-avatar">✦</div><div class="assistant-bubble"><p>${esc(message.text).replace(/\n/g,'<br>')}</p>${message.req?`<div class="assistant-constraints"><span>识别条件</span><b>${esc(aiConstraintText(message.req))}</b></div>`:''}${message.results?.length?`<div class="assistant-results">${message.results.map(aiResultCard).join('')}</div>`:''}</div></div>`}
  function saveAssistant(){writeStored('mcul_assistant_history',state.assistantMessages.slice(-12).map(message=>({role:message.role,text:message.text,req:message.req,results:(message.results||[]).map(item=>({id:item.device.id,score:item.score,reasons:item.reasons}))})))}
  function restoreAssistant(){return readStoredArray('mcul_assistant_history').filter(message=>message&&typeof message==='object').map(message=>({...message,req:message.req?{...message.req,coreAny:message.req.coreAny||[],excludedCores:message.req.excludedCores||[],softExcludedCores:message.req.softExcludedCores||[],excludedVendors:message.req.excludedVendors||[],minimums:message.req.minimums||{},maximums:message.req.maximums||{},excludedFeatures:message.req.excludedFeatures||[],softExcludedFeatures:message.req.softExcludedFeatures||[],vagueFeatures:message.req.vagueFeatures||[],flashArchitecture:message.req.flashArchitecture||[],ramTypes:message.req.ramTypes||[],ramTypeAny:message.req.ramTypeAny||[],technicalPreferences:message.req.technicalPreferences||[],technicalRequirements:message.req.technicalRequirements||[]}:null,results:(Array.isArray(message.results)?message.results:[]).filter(item=>item&&item.id).map(item=>{const device=byId.get(item.id);return device?{...item,device}:null}).filter(Boolean)}))}
  function renderAssistant(){
    const messages=state.assistantMessages.length?state.assistantMessages:[{role:'assistant',text:'告诉我你的资源约束，我会从当前离线目录中给出可核对的候选。',req:null,results:[]}];
    $('#view').innerHTML=`<div class="assistant-heading page-heading"><div><h1>选型助手 <span class="ai-badge">AI</span></h1><p>本地轻量模型 v${esc(localModel.version)} · 目录约束离线核验</p></div><button id="assistant-reset" class="assistant-reset" title="清空对话">清空</button></div><div class="assistant-shell"><div class="assistant-messages" id="assistant-messages">${messages.map(aiMessageHtml).join('')}</div><div class="assistant-quick"><button data-ai-prompt="需要 120MHz 以上、2 个 UART、CAN、64KB RAM 的 Cortex-M4">Cortex-M4 + CAN</button><button data-ai-prompt="需要 Wi-Fi、蓝牙和 USB，优先低功耗">Wi-Fi + 蓝牙</button><button data-ai-prompt="MicroPython，至少 2 个串口，带摄像头接口">MicroPython + 摄像头</button></div><form class="assistant-composer" id="assistant-form"><textarea id="assistant-input" rows="2" placeholder="例如：需要 120MHz、2 个 UART、CAN、64KB RAM 的 Cortex-M4"></textarea><button class="assistant-send" type="submit">生成候选 <span>↵</span></button></form></div>`;
    const form=$('#assistant-form'),input=$('#assistant-input'),messagesEl=$('#assistant-messages');form.onsubmit=e=>{e.preventDefault();const prompt=input.value.trim();if(!prompt)return;const result=aiRecommend(aiContextualPrompt(prompt));state.assistantMessages.push({role:'user',text:prompt},{role:'assistant',text:result.text,req:result.req,results:result.results});saveAssistant();renderAssistant()};input.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')form.requestSubmit()};$('#assistant-reset').onclick=()=>{state.assistantMessages=[];try{localStorage.removeItem('mcul_assistant_history')}catch(_){}renderAssistant()};document.querySelectorAll('[data-ai-prompt]').forEach(button=>button.onclick=()=>{input.value=button.dataset.aiPrompt;input.focus()});document.querySelectorAll('[data-ai-device]').forEach(card=>{card.onclick=()=>openDetail(card.dataset.aiDevice);card.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openDetail(card.dataset.aiDevice)}}});document.querySelectorAll('[data-ai-compare]').forEach(button=>button.onclick=e=>{e.stopPropagation();toggleCompare(button.dataset.aiCompare);button.textContent=state.compare.has(button.dataset.aiCompare)?'已对比':'＋ 对比';button.classList.toggle('selected',state.compare.has(button.dataset.aiCompare))});if(messagesEl)messagesEl.scrollTop=messagesEl.scrollHeight;updateNav();
  }
  function renderCompare(){
    const list=[...state.compare].map(id=>byId.get(id)).filter(Boolean);if(!list.length){$('#view').innerHTML='<div class="page-heading"><h1>参数对比</h1><p>最多同时比较四款器件。</p></div><div class="empty"><strong>尚未加入对比</strong>在器件详情或搜索结果中点击“对比”。</div>';updateNav();return}
    const numeric=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
    const factsFor=new Map(list.map(d=>[d.id,engineeringFacts(d)]));
    const facts=d=>factsFor.get(d.id)||{};
    const sumKnown=(...values)=>values.some(v=>numeric(v)!==null)?values.reduce((sum,v)=>sum+(numeric(v)??0),0):null;
    const runPowerBasis=commonPowerBasis(list,{mode:'run',typicalOnly:true})||commonPowerBasis(list,{mode:'run'});
    const sleepPowerBasis=commonPowerBasis(list,{mode:'sleep',typicalOnly:true})||commonPowerBasis(list,{mode:'sleep'});
    const powerDisplay=(d,mode,basis)=>basis?powerMetric(d,{mode,basis,typicalOnly:true}):null;
    const row=(label,display,metric,direction='max')=>{const scores=metric?list.map(metric).map(numeric):[];const known=scores.filter(v=>v!==null);const best=known.length>=2&&new Set(known).size>1?(direction==='min'?Math.min(...known):Math.max(...known)):null;return `<tr><td>${esc(label)}</td>${list.map((d,i)=>{const text=String(display(d)??'—');return `<td title="${esc(text)}" class="${best!==null&&scores[i]===best?'compare-best':''}">${esc(text)}</td>`}).join('')}</tr>`};
    $('#view').innerHTML=`<div class="page-heading"><h1>参数对比</h1><p>${list.length} / 4 款器件，绿色标出存在差异且数值领先的项目；“未核验”不会被当成“无”。</p></div><div class="compare-scroll"><table class="compare-table" style="--compare-columns:${list.length+1}"><thead><tr><th>项目</th>${list.map(d=>`<th><span class="compare-device-name" title="${esc(d.n)}">${esc(d.n)}</span><button class="remove-compare" data-remove="${esc(d.id)}">移出</button></th>`).join('')}</tr></thead><tbody>${row('厂商',d=>d.m)}${row('目录',d=>d.s+' › '+d.l)}${row('核心',d=>engineeringValue(d.c||d.a))}${row('FPU',d=>yesNoValue(d.fpu),d=>d.fpu==='yes'?1:d.fpu==='no'?0:null)}${row('DSP',d=>yesNoValue(d.dsp),d=>d.dsp==='yes'?1:d.dsp==='no'?0:null)}${row('MPU',d=>yesNoValue(d.mpu),d=>d.mpu==='yes'?1:d.mpu==='no'?0:null)}${row('TrustZone',d=>yesNoValue(d.tz),d=>d.tz==='yes'?1:d.tz==='no'?0:null)}${row('最高主频',d=>d.hz?clock(d.hz):'未核验',d=>d.hz)}${row('片上 Flash',d=>engineeringMemory(d.fl),d=>d.fl)}${row('片上 RAM',d=>engineeringMemory(d.ra),d=>d.ra)}${row('Flash 属性',d=>facts(d).flashFacts?.length?facts(d).flashFacts.join(' · '):'未核验')}${row('RAM 结构',d=>facts(d).ramRegions?.length?facts(d).ramRegions.join(' · '):'未核验')}${row('Cache',d=>facts(d).cache?'有':'未核验',d=>facts(d).cache?1:null)}${row('工作电压',voltageRange,d=>voltageRangeWidth(d))}${row('典型 / 运行功耗',d=>{const item=powerDisplay(d,'run',runPowerBasis);return item?powerValue(item.item):'未核验'},d=>{if(!runPowerBasis)return null;const item=powerMetric(d,{mode:'run',typicalOnly:true,basis:runPowerBasis});return item?.value??null},'min')}${row('睡眠 / 待机功耗',d=>{const item=powerDisplay(d,'sleep',sleepPowerBasis);return item?powerValue(item.item):'未核验'},d=>{if(!sleepPowerBasis)return null;const item=powerMetric(d,{mode:'sleep',typicalOnly:true,basis:sleepPowerBasis});return item?.value??null},'min')}${row('定时器位宽',d=>facts(d).timerWidths?.length?facts(d).timerWidths.join(' / ')+' bit':'未核验',d=>facts(d).timerWidths?.length?Math.max(...facts(d).timerWidths):null)}${row('TIM 总数',d=>engineeringValue(d.tim),d=>d.tim)}${row('ADC 单元',d=>engineeringValue(d.adcu),d=>d.adcu)}${row('ADC 通道（含内部）',d=>engineeringValue(d.adch),d=>d.adch)}${row('ADC 分辨率',d=>facts(d).adcResolution?facts(d).adcResolution+' bit':'未核验',d=>facts(d).adcResolution)}${row('ADC 采样率',d=>facts(d).adcRate?clock(facts(d).adcRate):'未核验',d=>facts(d).adcRate)}${row('DAC',d=>engineeringValue(d.dac),d=>d.dac)}${row('DAC 采样率',d=>facts(d).dacRate?clock(facts(d).dacRate):'未核验',d=>facts(d).dacRate)}${row('GPIO',d=>engineeringValue(d.gpio),d=>d.gpio)}${row('GPIO 速度',d=>facts(d).ioRate?clock(facts(d).ioRate):'未核验',d=>facts(d).ioRate)}${row('DMA 通道',d=>engineeringValue(d.dma),d=>d.dma)}${row('SERCOM / FLEXCOM',d=>engineeringValue(d.sercom),d=>d.sercom)}${row('SPI / I²C',d=>engineeringValue(d.spi)+' / '+engineeringValue(d.i2c),d=>sumKnown(d.spi,d.i2c))}${row('USART / UART',d=>engineeringValue(d.usart)+' / '+engineeringValue(d.uart),d=>sumKnown(d.usart,d.uart))}${row('CAN',d=>engineeringValue(d.can),d=>d.can)}${row('Flash 等待周期',d=>facts(d).flashFacts?.find(item=>/等待周期|零等待/.test(item))||'未核验')}${row('Flash ECC',d=>String(d.fecc||'').toLowerCase()==='yes'?'有':'未核验',d=>String(d.fecc||'').toLowerCase()==='yes'?1:null)}${row('RAM ECC',d=>String(d.recc||'').toLowerCase()==='yes'?'有':'未核验',d=>String(d.recc||'').toLowerCase()==='yes'?1:null)}${row('选型指数',d=>engineeringValue(d.idx)+' / 100',d=>d.idx)}${row('数据覆盖率',d=>engineeringValue(d.cov)+'%',d=>d.cov)}${row('完整订货号',d=>(d.parts||[]).length,d=>(d.parts||[]).length)}</tbody></table></div>`;document.querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{state.compare.delete(b.dataset.remove);saveCompare();renderCompare()});updateNav();
    const compareBody=$('.compare-table tbody');
    if(compareBody)compareBody.insertAdjacentHTML('beforeend',row('USB（通用 / Device / Host）',d=>engineeringValue(d.usb)+' / '+engineeringValue(d.usbd)+' / '+engineeringValue(d.usbh),d=>sumKnown(d.usb,d.usbd,d.usbh)));
    if(compareBody)compareBody.insertAdjacentHTML('beforeend',row('外部存储总线',d=>Array.isArray(d.eb)?((d.eb||[]).join(' / ')||'无'):'未核验'));
  }
  function renderData(){
    $('#view').innerHTML=`<div class="page-heading"><h1>数据与版本</h1><p>当前目录快照、可信度规则和厂商覆盖情况。</p></div><div class="info-banner"><b>离线快照 ${esc(catalog.meta.snapshot)}</b><p>全库评分字段平均覆盖率已通过 90% 构建门槛。缺失能力仍显示“—”，不会通过同系列型号自行补齐。</p></div><div class="summary-strip">${summary('平均覆盖率',catalog.meta.averageCoverage+'%')}${summary('覆盖率 ≥ 90%',count(catalog.meta.devicesAt90))}${summary('FPU 已核验',catalog.meta.fpuCoverage+'%')}</div><div class="summary-strip">${summary('系列大类',count(catalog.meta.series))}${summary('产品线',count(catalog.meta.productLines))}${summary('器件变体',count(catalog.meta.devices))}</div><div class="data-panel"><h3>覆盖范围</h3><table class="coverage-table"><thead><tr><th>厂商</th><th>系列</th><th>产品线</th><th>变体</th><th>订货号</th></tr></thead><tbody>${catalog.coverage.map(c=>`<tr><td>${esc(c.m)}</td><td>${count(c.series)}</td><td>${count(c.lines)}</td><td>${count(c.devices)}</td><td>${count(c.parts)}</td></tr>`).join('')}</tbody></table></div><div class="data-panel"><h3>可信度规则</h3><p class="good">● 完整订货号只收录有官方来源的记录，不通过后缀排列组合生成。</p><p>● MCUS 选型指数用于候选排序，不是 CoreMark、DMIPS 或 ULPMark 实测成绩。</p><p>● FPU 的“有 / 无”都必须来自处理器元数据、官方目标能力宏或明确的核心架构事实。</p><p>● 厂商加速器保留原名；没有逐器件完成确认的 Chrom-ART、Neural-ART 等标为待核验候选。</p><p class="caution">● 当前数据是已导入范围，不代表所有厂商完整在售目录已经全部完成。</p></div><div class="data-panel"><h3>应用版本</h3><p>MCUS Android ${esc(catalog.meta.version)} · 现代浅色界面版<br>生成时间：${esc(catalog.meta.generated)}</p></div>`;updateNav();
  }
  function renderAuthor(){
    $('#view').innerHTML=`<div class="page-heading"><h1>关于 MCUS</h1><p>一个面向工程师的离线 MCU 选型目录。</p></div><div class="author-card"><div class="chip-logo">M</div><h2>作者：new.bmp</h2><p>MCUS 汇总厂商 MCU、器件变体、外设资源、核心能力与官方订货号，帮助工程师快速筛选和比较。</p><p>当前版本：${esc(catalog.meta.version)} · 数据器件：${count(catalog.meta.devices)}</p><a class="author-link" href="https://github.com/new-bmp/MCUS">项目主页<br>https://github.com/new-bmp/MCUS ↗</a></div>`;
    updateNav();
  }
  function spec(v,label){const text=String(v??'—');return `<div class="spec-cell"><b title="${esc(text)}">${esc(text)}</b><span>${esc(label)}</span></div>`}
  function engineeringFacts(d){
    const items=(d?.pi||[]).map(item=>[item?.n,item?.d,item?.t].filter(Boolean).join(' '));
    const text=[...items,d?.acc,d?.feat,d?.pending].flat().filter(Boolean).join(' | ').toLowerCase();
    const widths=new Set();if(Number(d?.tw)>0)widths.add(Number(d.tw));
    for(const match of text.matchAll(/(?<!\d)(\d+)\s*-?\s*bit[^,;|]{0,24}(?:advanced\s+|general\s+|basic\s+)?(?:timer|counter|定时器|计数器)/gi))widths.add(Number(match[1]));
    for(const match of text.matchAll(/(?:timer|counter|定时器|计数器)[^,;|]{0,24}(?<!\d)(\d+)\s*-?\s*bit/gi))widths.add(Number(match[1]));
    const rate=(pattern)=>{const match=new RegExp(pattern+'[^,;|]{0,40}?(\\d+(?:\\.\\d+)?)\\s*(g|m|k)?(?:sps|samples?/s|mhz|khz|ghz)','i').exec(text)||new RegExp('(\\d+(?:\\.\\d+)?)\\s*(g|m|k)?(?:sps|samples?/s|mhz|khz|ghz)[^,;|]{0,40}?'+pattern,'i').exec(text);if(!match)return null;const n=Number(match[1]||match[2]);const unit=String(match[2]||'').toLowerCase();return n*(unit==='g'?1e9:unit==='m'?1e6:unit==='k'?1e3:1)};
    const adcResolution=Number(d?.adr)>0?Number(d.adr):((text.match(/(?<!\d)(\d+)\s*-?\s*bit[^,;|]{0,18}(?:adc|模数转换)/i)||text.match(/(?:adc|模数转换)[^,;|]{0,18}(?<!\d)(\d+)\s*-?\s*bit/i))?.[1]||null);
    const dacResolution=(text.match(/(?<!\d)(\d+)\s*-?\s*bit[^,;|]{0,18}(?:dac|数模转换)/i)||text.match(/(?:dac|数模转换)[^,;|]{0,18}(?<!\d)(\d+)\s*-?\s*bit/i))?.[1]||null;
    const directAdcRate=Number(d?.adcr)>0?Number(d.adcr):null;
    const directDacRate=Number(d?.dacr)>0?Number(d.dacr):null;
    const directIoRate=Number(d?.iospeed)>0?Number(d.iospeed):null;
    const directWait=d?.fw!==undefined&&d?.fw!==null&&d?.fw!==''&&Number.isFinite(Number(d.fw))?Number(d.fw):null;
    const directBanks=Number(d?.fb)>0?Number(d.fb):null;
    const flashFacts=[];
    if(directWait!==null)flashFacts.push(directWait===0?'零等待':`${directWait} 等待周期`);
    if(directBanks!==null)flashFacts.push(`${directBanks===2?'双':'单'} Bank`);
    if(String(d?.fecc||'').toLowerCase()==='yes')flashFacts.push('ECC');
    if(/zero[- ]?wait|0[- ]?wait|零等待/i.test(text))flashFacts.push('零等待');
    const wait=text.match(/(\d+)\s*(?:[- ]?wait(?:ing)?\s*states?|等待(?:周期|状态)?)/i);if(wait)flashFacts.push(`${wait[1]} 等待周期`);
    if(/dual[- ]?(?:bank|banked)|双\s*bank|双区|两区/i.test(text))flashFacts.push('双 Bank');
    if(/single[- ]?(?:bank|banked)|单\s*bank|单区/i.test(text))flashFacts.push('单 Bank');
    if(/hybrid|混合型|杂合型/i.test(text))flashFacts.push('混合型');
    if(/flash[^,;]{0,40}ecc|ecc[^,;]{0,40}flash/i.test(text))flashFacts.push('ECC');
    const ramRegions=(d?.mem||[]).filter(item=>/(?:ram|sram|tcm|ccm|psram|memory)/i.test(String(item?.n||''))&&!/(?:flash|rom|factory|nonmain)/i.test(String(item?.n||''))).map(item=>item.s?`${item.n} ${memory(item.s)}`:item.n).filter(Boolean);
    if(d?.ramarch)String(d.ramarch).split(';').filter(Boolean).forEach(item=>{if(!ramRegions.some(region=>String(region).toLowerCase().startsWith(item.toLowerCase())))ramRegions.push(item)});
    const ramTypes=[...new Set(ramRegions.map(item=>String(item).split(/\s+/)[0]).filter(Boolean))];
    if(/itcm/i.test(text)&&!ramTypes.some(item=>/itcm/i.test(item)))ramTypes.push('ITCM');
    if(/dtcm/i.test(text)&&!ramTypes.some(item=>/dtcm/i.test(item)))ramTypes.push('DTCM');
    if(/ccm/i.test(text)&&!ramTypes.some(item=>/ccm/i.test(item)))ramTypes.push('CCM');
    if(/axi\s*sram/i.test(text)&&!ramTypes.some(item=>/axi/i.test(item)))ramTypes.push('AXI SRAM');
    return {timerWidths:[...widths].filter(Number.isFinite).sort((a,b)=>a-b),adcResolution:Number(adcResolution)||null,dacResolution:Number(dacResolution)||null,adcRate:directAdcRate||rate('(?:adc|模数转换|采样)'),dacRate:directDacRate||rate('(?:dac|数模转换)'),ioRate:directIoRate||rate('(?:gpio|i/o|io|引脚翻转)'),flashFacts:[...new Set(flashFacts)],ramRegions,ramTypes,cache:String(d?.cache||'').toLowerCase()==='yes'||/i-cache|d-cache|icache|dcache|cache/i.test(text),ramEcc:String(d?.recc||'').toLowerCase()==='yes'||/ram[^,;]{0,24}ecc|ecc[^,;]{0,24}ram/i.test(text),dma:d?.dma,externalBus:Array.isArray(d?.eb)?d.eb:[],multiCore:Number(d?.cc)>1,exclusiveRam:String(d?.ramex||'').toLowerCase()==='yes'||/exclusive|private|dedicated|独占|专用|私有|per-core|每核/i.test(text)};
  }
  function voltageNumber(value){const numeric=Number(value);if(!Number.isFinite(numeric)||numeric<=0)return null;return numeric.toFixed(3).replace(/\.?0+$/,'')}
  function voltageRange(d){const min=voltageNumber(d?.vmin??d?.operatingVoltageMinV),max=voltageNumber(d?.vmax??d?.operatingVoltageMaxV);return min!==null&&max!==null&&Number(d?.vmax??d?.operatingVoltageMaxV)>=Number(d?.vmin??d?.operatingVoltageMinV)?`${min}-${max} V`:'—'}
  function voltageRangeWidth(d){const min=Number(d?.vmin??d?.operatingVoltageMinV),max=Number(d?.vmax??d?.operatingVoltageMaxV);return Number.isFinite(min)&&Number.isFinite(max)&&max>=min?max-min:null}
  function powerModeLabel(mode){return {run:'运行 / 活动',sleep:'睡眠 / 待机',other:'其他模式'}[String(mode||'').toLowerCase()]||String(mode||'其他模式')}
  function powerQualityLabel(quality){return {typical:'典型值',maximum:'最大值',minimum:'最小值'}[String(quality||'').toLowerCase()]||'来源未注明典型 / 最大'}
  function powerValue(item){const raw=Number(item?.v);if(!Number.isFinite(raw))return '—';const formatted=raw.toFixed(6).replace(/\.?0+$/,'');return `${formatted} ${String(item?.u||'')}`.trim()}
  function powerConditions(item){const conditions=item&&typeof item.c==='object'?item.c:{},parts=[];if(Number(conditions.hz)>0)parts.push(clock(Number(conditions.hz)));if(Number(conditions.v)>0)parts.push(Number(conditions.v)+' V');if(Number.isFinite(Number(conditions.t)))parts.push(Number(conditions.t)+' °C');if(conditions.n)parts.push(String(conditions.n));return parts.length?parts.join(' · '):'条件未完整披露'}
  function powerMeasurements(d){return Array.isArray(d?.pwr)?d.pwr.filter(item=>item&&Number.isFinite(Number(item.v))&&item.u):[]}
  function powerSection(d){
    const items=powerMeasurements(d);
    if(!items.length)return detailAccordion('功耗与电源','<div class="feature-panel"><div class="feature-label">当前官方来源没有可比较的典型功耗或电流测量值。模式名称、模式数量和无单位参数不会被当作功耗。</div></div>',false,'未核验');
    const body=`<div class="inventory-note"><b>功耗必须连同测试条件一起看。</b> 仅展示来源中带明确 A/W 单位的测量；不同电压、主频、温度和外设状态不能直接比较。</div><div class="power-grid">${items.map(item=>{const label=item.l||powerModeLabel(item.m),conditions=powerConditions(item);return `<details class="power-card"><summary title="${esc(label)}"><div><span>${esc(powerModeLabel(item.m))}</span><em>${esc(powerQualityLabel(item.q))}</em></div><b>${esc(powerValue(item))}</b><p>${esc(label)}</p><i class="power-chevron" aria-hidden="true">⌄</i></summary><div class="power-card-detail"><p>${esc(label)}</p>${conditions?`<small>${esc(conditions)}</small>`:''}</div></details>`}).join('')}</div>`;
    const typical=items.filter(item=>item.q==='typical').length;
    return detailAccordion('功耗与电源',body,false,typical?`${typical} 项典型值`:`${items.length} 项测量`);
  }
  function powerUnitBasis(unit){const normalized=String(unit||'').replace(/μ|µ/g,'u').toLowerCase();if(normalized.endsWith('_per_mhz'))return 'current_per_mhz';if(normalized.endsWith('a'))return 'current';if(normalized.endsWith('w'))return 'power';return ''}
  function powerNormalized(value,unit){const numeric=Number(value),normalized=String(unit||'').replace(/μ|µ/g,'u').toLowerCase();if(!Number.isFinite(numeric))return null;const factors={a:1e6,ma:1e3,ua:1,na:.001,w:1e6,mw:1e3,uw:1,'ma_per_mhz':1e3,'ua_per_mhz':1};return Object.prototype.hasOwnProperty.call(factors,normalized)?numeric*factors[normalized]:null}
  function powerMetric(d,request={}){const mode=request.mode||'',requestedBasis=request.basis||'',typicalOnly=Boolean(request.typicalOnly);const source=powerMeasurements(d).filter(item=>(!mode||item.m===mode)&&(!typicalOnly||item.q==='typical'));const bases=requestedBasis?[requestedBasis]:['current','power','current_per_mhz'];for(const basis of bases){const candidates=source.filter(item=>powerUnitBasis(item.u)===basis).map(item=>({item,value:powerNormalized(item.v,item.u)})).filter(entry=>entry.value!==null);if(candidates.length)return candidates.sort((a,b)=>a.value-b.value)[0]}return null}
  function commonPowerBasis(list,request={}){const bases=['current','power','current_per_mhz'];return bases.find(basis=>list.filter(d=>powerMetric(d,{...request,basis})).length>=2)||null}
  function detailAccordion(title,content,open=false,meta=''){
    return `<details class="detail-section detail-accordion"${open?' open':''}><summary><span>${esc(title)}</span>${meta?`<em>${esc(meta)}</em>`:''}<b class="accordion-chevron" aria-hidden="true">⌄</b></summary><div class="detail-section-body">${content}</div></details>`;
  }
  function inventorySection(d){
    const items=d.pi||[];
    const categoryLabels={timing:'定时与控制',analog:'模拟外设',gpio:'GPIO 与中断',connectivity:'通信接口',wireless:'无线连接',memory_bus:'DMA 与外部总线',display_multimedia:'显示与多媒体',security:'安全',accelerator:'计算加速',clock:'时钟',power:'电源与低功耗',system:'系统资源',other:'其他来源特征'};
    const order=['timing','analog','gpio','connectivity','wireless','memory_bus','display_multimedia','security','accelerator','clock','power','system','other'];
    if(!items.length)return detailAccordion('来源外设清单','<div class="feature-panel"><div class="feature-label">当前来源没有可展开的外设特征；不代表芯片没有外设。</div></div>',false,'暂无记录');
    const grouped=new Map();items.forEach(item=>{const key=item.g||'other';if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(item)});
    const inventoryItem=item=>{const name=String(item?.n||'未命名资源'),detail=String(item?.d||'').trim();return `<details class="inventory-item"><summary title="${esc(name)}"><b>${esc(name)}</b><i class="inventory-item-chevron" aria-hidden="true">⌄</i></summary>${detail?`<p>${esc(detail)}</p>`:''}</details>`};
    return detailAccordion('来源外设清单',`<div class="inventory-note">这里逐项展示来源明确列出的资源。ADC 以转换器单元和通道为选型参数，不统计 ADC 引脚数量。</div>${order.filter(key=>grouped.has(key)).map(key=>`<details class="inventory-group"><summary><span>${esc(categoryLabels[key]||key)}</span><em>${grouped.get(key).length} 项</em><b class="accordion-chevron" aria-hidden="true">⌄</b></summary><div class="inventory-list">${grouped.get(key).map(inventoryItem).join('')}</div></details>`).join('')}`,false,`${items.length} 项`);
  }
  function packageNames(d){
    const names=String(d.pkg||'').split(/[;,]/).map(item=>item.trim()).filter(Boolean);
    if(names.length)return [...new Set(names)];
    return [...new Set((d.parts||[]).map(part=>String(part.p||'').trim()).filter(Boolean))];
  }
  function packageEntries(d){
    const names=packageNames(d),pins=String(d.pin||'').split(/[;,]/).map(item=>Number(item.trim())).filter(item=>Number.isFinite(item)&&item>0);
    if(!names.length)return [];
    if(names.length===1&&pins.length>1)return pins.map(pin=>({name:names[0],pins:pin}));
    return names.map((name,index)=>{
      let pin=pins[index];
      if(!Number.isFinite(pin)&&names.length===1&&pins.length===1)pin=pins[0];
      if(!Number.isFinite(pin)){const match=/(?:QFN|QFP|LQFP|TQFP|UFQFPN|UFBGA|LFBGA|BGA|LGA|WLCSP|CSP|SOIC|SOP|SSOP|TSSOP|MSOP|DFN|DIP|PDIP|PLCC)[^0-9]{0,3}(\d{2,4})/i.exec(name);pin=match?Number(match[1]):null}
      return {name,pins:Number.isFinite(pin)&&pin>0?pin:null};
    });
  }
  function packageKind(name){
    const value=String(name||'').toUpperCase();
    if(/DIP|PDIP|SIP|ZIP/.test(value))return 'through-hole';
    if(/BGA|FBGA|LGA|WLCSP|CSP|TFLGA/.test(value))return 'array';
    if(/QFN|DFN|UFQFPN|HVQFN|VQFN/.test(value))return 'leadless';
    if(/QFP|LQFP|TQFP|UQFP|PQFP/.test(value))return 'qfp';
    if(/SOIC|SOP|SSOP|TSSOP|MSOP|TSOP|SOT/.test(value))return 'gullwing';
    if(/RF MODULE/.test(value))return 'module';
    return 'generic';
  }
  function packagePins(count,side){
    if(!count)return '';
    const sideIndex={top:0,right:1,bottom:2,left:3}[side]??0;
    const sideCount=Math.min(64,Math.floor(count/4)+(sideIndex<count%4?1:0));
    if(!sideCount)return '';
    return `<div class="package-pins ${side}">${Array.from({length:sideCount},()=>'<i></i>').join('')}</div>`;
  }
  function documentLabel(doc){return String(doc.title||doc.name||doc.path||'').trim()}
  function isDatasheetDocument(doc){
    const kind=String(doc.kind||'').toLowerCase();
    return kind==='datasheet'||/(?:\bdata\s*-?\s*sheet\b|\bdatasheet\b|数据手册)/i.test(documentLabel(doc));
  }
  function isManualDocument(doc){
    const kind=String(doc.kind||'').toLowerCase();
    if(new Set(['datasheet','reference_manual','user_manual','technical_manual','manual']).has(kind))return true;
    return /(?:\bdata\s*-?\s*sheet\b|\bdatasheet\b|\breference\s+manual\b|\buser\s+manual\b|\btechnical\s+manual\b|数据手册|参考手册|用户手册|技术手册)/i.test(documentLabel(doc));
  }
  function packageSection(d){
    const entries=packageEntries(d);
    const drawings=(Array.isArray(d.docs)?d.docs:[]).filter(doc=>doc.kind==='package_drawing'&&safeHttpUrl(doc.url));
    const datasheets=(Array.isArray(d.docs)?d.docs:[]).filter(doc=>isDatasheetDocument(doc)&&safeHttpUrl(doc.url));
    const packageDocs=drawings.length?drawings:datasheets.slice(0,2);
    const packageDocsTitle=drawings.length?'厂商封装图':'厂商数据手册中的封装尺寸资料';
    const drawingBlock=packageDocs.length?`<h3 class="document-group-title">${packageDocsTitle} · ${packageDocs.length}</h3>${documentRows(packageDocs)}`:'';
    if(!entries.length){const message=d.verify==='official_datasheet_variant_not_listed'?'该旧型号未出现在厂商当前数据手册的订货编码表中，原有推测封装已移除，等待厂商历史资料核验。':'当前官方来源没有提供可核验的封装名称。';return detailAccordion('封装图',`<div class="feature-panel"><div class="feature-label">${esc(message)}</div></div>${drawingBlock}`,false,'未确认')}
    return detailAccordion('封装图',`<div class="inventory-note">应用内示意图仅用于快速辨认；焊盘尺寸、引脚定义和包装尺寸以厂商封装图或数据手册为准。</div><div class="package-grid">${entries.map(entry=>{const kind=packageKind(entry.name);const packageLabel=entry.pins?`${entry.name} · ${entry.pins}`:entry.name;return `<div class="package-card"><div class="package-figure package-${kind}"><div class="package-body"><strong>${esc(d.n||'MCU')}</strong><small>${esc(packageLabel)}</small></div></div></div>`}).join('')}</div>${drawingBlock}`,false,`${entries.length} 种`);
  }
  function documentStatus(doc){
    const status=String(doc.status||'').toLowerCase(),http=Number(doc.http||0);
    if(http===404||http===410||status==='invalid')return {label:'已失效',className:'invalid'};
    if(status.includes('rate')||http===429)return {label:'厂商限流',className:'limited'};
    if(status.includes('waf')||http===403)return {label:'厂商防护',className:'limited'};
    if(status==='valid'||status.startsWith('official_')||http===200)return {label:'已核验',className:'valid'};
    if(doc.path&&!doc.url)return {label:'Pack 内文件',className:'local'};
    return {label:doc.url?'官方链接':'来源记录',className:'neutral'};
  }
  function documentRows(items){
    return `<div class="document-list">${items.map(doc=>{const url=safeHttpUrl(doc.url);const title=doc.title||doc.name||doc.path||'官方资料';const status=documentStatus(doc);const detail=[doc.version?`版本 ${doc.version}`:'',doc.path?`Pack 路径 ${doc.path}`:''].filter(Boolean).join(' · ');const body=`<span class="document-icon">${url?'↗':'DOC'}</span><span class="document-main"><b>${esc(title)}</b>${detail?`<small>${esc(detail)}</small>`:''}</span><span class="document-status ${status.className}">${esc(status.label)}</span>`;return url?`<a class="document-row" href="${esc(url)}" target="_blank" rel="noopener">${body}</a>`:`<div class="document-row document-local">${body}</div>`}).join('')}</div>`;
  }
  function documentsSection(d){
    const docs=Array.isArray(d.docs)?d.docs:[];
    const sourceUrls=safeHttpUrls(d.src);
    const manuals=docs.filter(isManualDocument);
    const sources=docs.filter(doc=>!isManualDocument(doc)&&doc.kind!=='package_drawing');
    sourceUrls.forEach((sourceUrl,index)=>{if(!sources.some(doc=>safeHttpUrl(doc.url)===sourceUrl)&&!manuals.some(doc=>safeHttpUrl(doc.url)===sourceUrl))sources.push({title:sourceUrls.length>1?`官方来源页面 ${index+1}`:'官方来源页面',url:sourceUrl,kind:'source'})});
    const manualBlock=manuals.length?documentRows(manuals):'<div class="feature-panel"><div class="feature-label">当前来源没有可直接打开的手册；Pack 内相对路径不会伪装成网页链接。</div></div>';
    const sourceBlock=sources.length?`<h3 class="document-group-title">产品页与器件包来源 · ${sources.length}</h3>${documentRows(sources)}`:'';
     return detailAccordion('官方手册',`${manualBlock}${sourceBlock}<p class="score-note">“已核验”表示链接审计可访问或来自厂商官方文档接口；“厂商限流/防护”不等于链接失效。</p>`,false,`${manuals.length} 份`);
  }
  function isUsableManualDocument(doc){
    if(!doc||!safeHttpUrl(doc.url))return false;
    const status=String(doc.status||doc.verification_status||'').toLowerCase();
    if(status==='invalid'||status.includes('limited')||status.includes('rate')||status.includes('waf'))return false;
    return isManualDocument(doc)||/\.pdf(?:$|[?#])/i.test(String(doc.url));
  }
  function missingInfoReasons(d){
    const reasons=[];
    const docs=Array.isArray(d?.docs)?d.docs:[];
    if(!docs.some(isUsableManualDocument))reasons.push('没有可用手册');
    if(!packageNames(d).length)reasons.push('封装未核验');
    if(!String(d?.c||d?.a||'').trim())reasons.push('内核未核验');
    if(!(Number(d?.hz)>0))reasons.push('主频未核验');
    if(!(Number(d?.fl)>0))reasons.push('Flash 未核验');
    if(!(Number(d?.ra)>0))reasons.push('RAM 未核验');
    if(!(Array.isArray(d?.pi)&&d.pi.length))reasons.push('外设资源未核验');
    return reasons;
  }
  function missingInfoLevel(reasons){
    return reasons.length===1&&reasons[0]==='没有可用手册'?'manual':'critical';
  }
  function missingInfoHeaderClass(reasons){
    if(!reasons.length)return '';
    return missingInfoLevel(reasons)==='manual'?'detail-header-warning-manual':'detail-header-warning';
  }
  function missingInfoBadge(d){
    const reasons=missingInfoReasons(d);if(!reasons.length)return '';
    const tone=missingInfoLevel(reasons)==='manual'?' manual-warning':'';
    return `<i class="data-warning${tone}" role="img" aria-label="关键资料缺失" title="${esc(reasons.join('；'))}">⚠</i>`;
  }
  const acceleratorGlossary=[
    {pattern:/chrom[\s-]*art|dma2d/i,title:'Chrom-ART / DMA2D',kind:'2D 图形加速',description:'用于图像搬运、区域填充、颜色格式转换和图层混合，可减少 CPU 参与显示刷新。'},
    {pattern:/neo[\s-]*chrom(?:\s+vg)?/i,title:'NeoChrom VG',kind:'图形加速',description:'厂商图形引擎名称，面向矢量图形、图层和显示合成。它与 Chrom-ART 都能减少 CPU 的像素搬运，但支持的图元、格式和带宽不同，不能默认互相替代。'},
    {pattern:/gfxmmu|gpu2d|2d[\s-]*gpu|graphics? accelerator|图形加速/i,title:'2D 图形引擎',kind:'图形加速',description:'用于图层、像素格式、裁剪或图形合成的专用硬件；具体图元、分辨率和带宽以本型号手册为准。'},
    {pattern:/neural[\s-]*art|neural|npu|ai[\s-]*accelerator|神经网络|ai加速/i,title:'Neural / AI 加速器',kind:'机器学习加速',description:'面向神经网络或矩阵运算的专用硬件。模型算子、片上存储、量化格式和吞吐限制必须以该型号手册为准。'},
    {pattern:/hsp[\s-]*1/i,title:'HSP1',kind:'厂商专用处理模块',description:'原厂 HSP1 模块名称，属于该系列的专用高速信号或数据处理资源。输入输出、连接方式和适用场景请以对应型号手册为准。'},
    {pattern:/cordic/i,title:'CORDIC',kind:'数学加速',description:'用迭代算法硬件加速三角函数、向量旋转、开方等定点数学运算，适合电机控制和信号处理。'},
    {pattern:/\bfmac\b|filter accelerator|滤波|乘加/i,title:'FMAC / 滤波乘加',kind:'滤波与乘加加速',description:'硬件乘加/滤波单元，可把常见 FIR、IIR 或矩阵乘加运算从 CPU 中卸载；数据格式和采样速率以手册为准。'},
    {pattern:/pka|ecc|rsa|aes|sha|hash|crypto|cryptograph|加密|安全引擎/i,title:'密码学加速器',kind:'安全加速',description:'用于 AES、SHA、ECC、RSA 等密码运算，降低软件实现的延迟和功耗；实际算法集合、密钥长度和安全边界以型号安全章节为准。'},
    {pattern:/rng|trng|true random|随机数/i,title:'RNG / TRNG',kind:'硬件随机数',description:'硬件随机数发生器为协议、密钥和安全启动提供随机源；熵源质量、健康检测和接口细节以手册为准。'},
    {pattern:/crc|循环冗余/i,title:'CRC',kind:'数据校验',description:'循环冗余校验硬件，用于快速检测通信帧和存储数据的传输错误，支持的多项式和位宽以手册为准。'},
    {pattern:/fmc|fsmc/i,title:'FMC / FSMC',kind:'外部存储控制器',description:'Flexible Memory Controller（FMC）或 Flexible Static Memory Controller（FSMC），用于连接并行 SRAM、NOR/NAND、SDRAM/PSRAM 等外部存储器。总线宽度、时序、存储器类型和 DMA 配合能力以本型号手册为准。'},
    {pattern:/octospi|octo[\s-]*spi|\bospi\b/i,title:'OCTOSPI / OSPI',kind:'八线串行存储接口',description:'支持八线/双四线串行存储器的高速接口，通常用于外部 NOR Flash、NAND Flash 或 HyperRAM。它不是片上 Flash 容量；线数、DDR、内存映射、DQS 和最高速率以本型号手册为准。'},
    {pattern:/quadspi|\bqspi\b/i,title:'QUADSPI / QSPI',kind:'四线串行存储接口',description:'支持单线、双线或四线 SPI Flash 的专用存储接口，可提供内存映射读取和 DMA 传输。具体命令、时钟、DDR 和片选数量以本型号手册为准。'},
    {pattern:/flexspi/i,title:'FlexSPI',kind:'可配置串行存储接口',description:'可配置的高速串行存储控制器，常用于 NOR Flash、HyperFlash 或 PSRAM；协议、端口、采样边沿和 LUT 配置以本型号手册为准。'},
    {pattern:/hyperbus/i,title:'HyperBus',kind:'高速外部存储总线',description:'面向 HyperRAM/HyperFlash 的高速低引脚数存储总线；总线宽度、时钟、内存映射和读写时序以本型号手册为准。'},
    {pattern:/emif|\bebi\b|\bsmc\b|\bsqi\b/i,title:'EMIF / EBI / SMC',kind:'外部存储接口',description:'厂商外部存储接口控制器，用于连接异步存储器、SRAM、NOR/NAND 或其他并行设备；支持的协议、地址宽度和时序以本型号手册为准。'},
    {pattern:/dma|direct memory|dmac|dtc/i,title:'DMA / DTC',kind:'数据搬运加速',description:'在外设与存储器之间搬运数据，支持无 CPU 或低 CPU 占用的数据流处理；通道数、触发源和寻址限制以手册为准。'},
    {pattern:/jpeg|jpg|h[.]?264|h[.]?265|codec|编解码/i,title:'媒体编解码器',kind:'图像 / 视频加速',description:'为 JPEG、视频或其他媒体格式提供硬件编解码能力，减少图像处理的 CPU 占用；支持的档次和分辨率以手册为准。'},
    {pattern:/camera|dcmi|pssi|dvp|摄像头|图像输入/i,title:'摄像头 / 图像输入',kind:'图像接口',description:'接收摄像头或并行图像数据的硬件接口，负责同步、采样和 DMA 传输；电气时序和像素格式以手册为准。'},
    {pattern:/ltdc|gfx|lcd|glcd|display|显示|图层/i,title:'显示控制器',kind:'显示接口',description:'负责显示时序、帧缓冲或图层输出，可将刷新工作从 CPU 中分离；分辨率、层数和像素格式以手册为准。'},
    {pattern:/ethernet|gigabit|eth|sata|以太网/i,title:'Ethernet / 高速网络',kind:'网络接口',description:'提供以太网 MAC、PHY 配套或高速网络接口；速率、DMA 描述符和外部 PHY 要求以手册为准。'},
    {pattern:/usb[\s-]*(?:super.?speed|ss)|superspeed/i,title:'USB SuperSpeed',kind:'USB 3.x 高速接口',description:'USB SuperSpeed 高速链路，带宽高于 USB 2.0；控制器角色、PHY、供电和具体速率以手册为准。'},
    {pattern:/usb[\s-]*pd|power delivery/i,title:'USB Power Delivery',kind:'USB 供电协商',description:'用于 USB-C 电源角色和电压电流协商的硬件支持；协议版本、功率档位和保护机制以手册为准。'},
    {pattern:/usb|通用串行总线/i,title:'USB 控制器',kind:'USB 接口',description:'提供 USB 设备、主机或 OTG 控制器能力；角色、端点数量、速率和 PHY 配置以本型号手册为准。'},
    {pattern:/i3c/i,title:'I3C',kind:'高速串行总线',description:'兼容 I²C 的双线高速总线，支持动态地址、带内中断和更高吞吐；目标设备兼容性以手册为准。'},
    {pattern:/sdmmc|sdio/i,title:'SDMMC / SDIO',kind:'存储卡接口',description:'用于连接 SD、SDIO、eMMC 或类似多线存储设备；总线宽度、主从角色、时钟和 DMA 能力以本型号手册为准。'},
    {pattern:/mipi/i,title:'MIPI',kind:'显示 / 摄像头高速接口',description:'用于 MIPI 显示或摄像头链路的高速串行接口；具体是 DSI、CSI、D-PHY 还是其他子协议，以及通道数和速率以本型号手册为准。'},
    {pattern:/flexray/i,title:'FlexRay',kind:'汽车实时网络',description:'面向汽车控制的确定性高速总线，支持时间触发通信和冗余链路；节点、缓冲区和收发器要求以本型号手册为准。'},
    {pattern:/can[\s-]*fd|canfd/i,title:'CAN FD',kind:'汽车控制总线',description:'CAN 的可变速率数据段扩展，支持比经典 CAN 更长的数据帧和更高数据速率；仲裁速率、数据速率和过滤器数量以本型号手册为准。'},
    {pattern:/opamp|operational amplifier|运算放大器/i,title:'OPAMP',kind:'模拟信号调理',description:'片上运算放大器，可用于传感器信号缓冲、放大、滤波或内部模拟通路；输入范围、增益带宽和可路由引脚以本型号手册为准。'},
    {pattern:/comparator|比较器|acmp|\bcomp\b/i,title:'比较器',kind:'模拟比较',description:'比较模拟输入与参考电压并输出数字状态，适合过零、窗口检测和快速保护；输入通道、迟滞和内部参考配置以本型号手册为准。'},
    {pattern:/sai|i2s|数字音频|audio/i,title:'SAI / I²S',kind:'数字音频接口',description:'用于音频采样、播放和多通道串行传输；时钟模式、数据槽位和采样率以手册为准。'},
    {pattern:/qei|qeo|hall|encoder|霍尔|编码器/i,title:'电机位置接口',kind:'电机控制',description:'用于增量编码器、霍尔传感器或电机换相位置采集；输入滤波、计数宽度和输出波形以手册为准。'},
    {pattern:/pio|state machine/i,title:'PIO / 可编程 IO',kind:'可编程外设',description:'由用户编程的 IO 状态机或时序引擎，可实现非标准串行协议和精确波形；指令数、状态机和 FIFO 资源以手册为准。'},
    {pattern:/psram|sram|itcm|dtcm|tcm|独占 ram|shared ram|共享 ram/i,title:'片上 RAM 结构',kind:'存储架构',description:'该能力描述片上 RAM、TCM、共享 RAM 或外部 PSRAM 的结构属性；容量、总线归属和多核访问规则以本型号手册为准。'},
    {pattern:/trustzone|secure boot|安全启动|flash encryption|安全存储/i,title:'安全启动 / TrustZone',kind:'系统安全',description:'用于启动链验证、隔离安全世界或保护片上存储；密钥生命周期、调试锁和安全边界以手册为准。'},
    {pattern:/touch|capsense|ptc|tsc|触摸|电容/i,title:'电容触摸检测',kind:'触摸接口',description:'利用电容变化检测触摸按键或滑条；通道数、灵敏度和校准方式以手册为准。'},
    {pattern:/serdes|hs?pi|高速串行/i,title:'高速串行收发器',kind:'高速接口',description:'用于高速串行数据收发或专用外设连接；线速、编码、通道数和信号完整性要求以手册为准。'},
    {pattern:/dfsdm|mdf|pdm|数字滤波|麦克风/i,title:'数字滤波 / 音频采集',kind:'信号处理外设',description:'用于数字麦克风、过采样 ADC 或多通道传感器数据的抽取、滤波和整流；滤波器数量、输入接口和采样率以本型号手册为准。'},
    {pattern:/ucpd|usbpd|power delivery|usb-c/i,title:'USB-C PD / UCPD',kind:'USB 供电协商',description:'用于 USB Type-C 连接检测和 Power Delivery 电源角色协商；协议版本、功率路径和保护功能以本型号手册为准。'},
    {pattern:/dmamux|dmas*channel|linkedlist|linked[s-]*list/i,title:'DMA 请求路由 / 链表',kind:'数据搬运控制',description:'DMAMUX 负责把外设请求路由到 DMA；链表队列可连续执行多段搬运，减少中断和 CPU 介入。请求源、通道数和队列限制以手册为准。'},
    {pattern:/ramecc|flexramecc|rams*ecc/i,title:'RAM ECC',kind:'存储可靠性',description:'为片上 RAM 提供错误检测或纠正，适合安全关键和高可靠应用。覆盖的 RAM 区域、纠错位数和故障注入能力以手册为准。'},
    {pattern:/icache|dcache|cache/i,title:'指令 / 数据缓存',kind:'存储架构',description:'片上缓存减少处理器访问 Flash 或共享存储的等待；容量、行大小、替换策略和一致性规则以手册为准。'},
    {pattern:/otfdec|on[s-]*the[s-]*fly|flashs*decrypt/i,title:'在线 Flash 解密',kind:'安全存储',description:'在取指或读取路径上对外部/片上加密内容解密，避免明文固件长期暴露；密钥、地址范围和启动链规则以手册为准。'},
    {pattern:/gtzc|sau|idau|secures*attribution/i,title:'安全区域控制',kind:'系统安全',description:'用于 TrustZone-M 的安全/非安全地址和外设访问归属控制；安全边界、默认属性和锁定行为以手册为准。'},
    {pattern:/hsem|hardwares*semaphore/i,title:'硬件信号量',kind:'多核 / 共享资源同步',description:'提供硬件级互斥标志，协调多核或安全域访问共享 RAM、外设和关键寄存器；信号量数量和中断行为以手册为准。'},
    {pattern:/evsys|events*system/i,title:'事件系统',kind:'外设互连',description:'在外设之间直接传递事件，减少 CPU 中断和软件轮询；通道数、触发源和用户映射以手册为准。'},
    {pattern:/ccl|configurables*customs*logic/i,title:'可配置逻辑 CCL',kind:'硬件逻辑',description:'用查找表和组合/时序逻辑实现定制门控、波形整形或外设互连；LUT 数量、输入源和时钟限制以手册为准。'},
    {pattern:/pdec|positions*decoder/i,title:'位置解码器',kind:'电机控制',description:'硬件解码增量编码器或方向脉冲并累计位置，通常支持滤波和索引输入；计数宽度、模式和输入引脚以手册为准。'},
    {pattern:/divas|division|maths*accelerator/i,title:'DIVAS 数学加速',kind:'数学运算',description:'为除法、平方根或定点数学提供专用运算路径，降低软件库开销；支持的数据宽度和异常行为以手册为准。'},
    {pattern:/vrefbuf|voltages*references*buffer/i,title:'内部参考电压缓冲',kind:'模拟参考',description:'把内部参考电压缓冲到 ADC、DAC、比较器或外部引脚；参考档位、驱动能力和稳定时间以手册为准。'},
    {pattern:/sdadc|sigma[s-]*delta/i,title:'Sigma-Delta ADC',kind:'高分辨率模拟',description:'利用过采样和数字滤波获得高分辨率测量，适合电流、压力和音频等慢速信号；采样率、滤波器和输入范围以手册为准。'},
    {pattern:/subghz|sub[s-]*ghz/i,title:'Sub-GHz 无线',kind:'低功耗无线',description:'面向低于 1 GHz 的远距离、低功耗无线链路；频段、调制、发射功率和协议栈支持以型号资料为准。'},
    {pattern:/bluetooth|\bble\b|zigbee|ieee802154/i,title:'无线协议硬件',kind:'无线连接',description:'来源确认的 Bluetooth LE、Zigbee 或 IEEE 802.15.4 无线能力；射频频段、协议版本和外部匹配要求以手册为准。'},
    {pattern:/hdmi[_\s-]*cec/i,title:'HDMI-CEC',kind:'影音连接',description:'用于 HDMI 设备间的控制命令和待机唤醒通信；电气电平、消息过滤和引脚复用以手册为准。'},
    {pattern:/swpmi|single[s-]*wire/i,title:'单线协议接口',kind:'车载 / 专用通信',description:'面向单线外设或车载节点的专用通信控制器；帧格式、速率、收发器和唤醒条件以手册为准。'},
    {pattern:/cap|qii|isp|bus8|ledpwm|mcpwm|bc\b|专用/i,title:'厂商专用能力',kind:'原厂特性',description:'该名称是厂商资料中确认的专用硬件能力。这里保留原始名称，具体用途、资源数量和限制请打开本型号官方手册核对。'},
    {pattern:/vdd|电压|low[\s-]*power|低功耗/i,title:'供电 / 低功耗特性',kind:'电源特性',description:'描述官方工作电压、低功耗模式或电源管理资源，不代表额外的计算加速；电流、唤醒源和限制以手册为准。'},
    {pattern:/core|cortex|qingke|arm|risc[\s-]*v|架构/i,title:'处理器核心 / 架构',kind:'核心能力',description:'该项是厂商资料确认的处理器核心或指令集架构。异常级别、扩展指令、FPU 和调试能力以本型号手册为准.'},
  ];
  function featureLabel(item){
    if(item&&typeof item==='object')return String(item.name||item.n||item.title||item.label||item.feature_id||'').trim();
    return String(item||'').trim();
  }
  function acceleratorEntry(item,status){
    const label=featureLabel(item);
    const type=String(item&&typeof item==='object'?item.type:'');
    const info=acceleratorGlossary.find(entry=>entry.pattern.test(label)||type&&entry.pattern.test(type));
    return {label,title:info?.title||label||'未命名能力',kind:info?.kind||'原厂专用能力',description:info?.description||'该能力已由来源资料确认包含在此型号中。由于厂商命名没有统一标准，具体用途、数量、接口关系和性能限制请以本型号官方手册为准。',status};
  }
  function capabilityLabels(item){
    const label=featureLabel(item);
    if(String(item&&typeof item==='object'?item.type:'')==='ExtBus'){
      const tokens=label.match(/\b(?:FMC|FSMC|OCTOSPI\d*|OCTOSPIM|OSPI\d*|XSPI\d*|XSPIM|QUADSPI\d*|QSPI\d*|FLEXSPI\d*|HYPERBUS|EMIF\d*|EBI\d*|SMC\d*|SQI\d*)\b/gi);
      if(tokens&&tokens.length)return [...new Set(tokens.map(token=>token.toUpperCase()))];
    }
    return label?[label]:[];
  }
  function acceleratorGrid(items){
    if(!items.length)return '<div class="feature-panel"><div class="feature-label">当前来源没有已确认的专用加速器或能力记录</div></div>';
    return `<div class="accelerator-grid">${items.map((item,index)=>`<article class="accelerator-card" data-accelerator-card><button class="accelerator-toggle" type="button" data-accelerator-toggle="${index}" aria-expanded="false"><span class="accelerator-code">${esc(item.title)}</span><span class="accelerator-kind">${esc(item.kind)}</span><span class="accelerator-chevron" aria-hidden="true">⌄</span></button><div class="accelerator-explanation" data-accelerator-explanation hidden><b>${esc(item.label)}</b><p>${esc(item.description)}</p><small>${item.status==='pending'?'待逐器件核验':'来源已确认；通用说明仅作选型参考'}</small></div></article>`).join('')}</div>`;
  }
  function releaseDateText(d){
    const year=String(d.ry||d.releaseYear||'').trim(),quarter=String(d.rq||d.releaseQuarter||'').trim().toUpperCase();
    if(year&&quarter)return `${year}/${/^Q?\d$/i.test(quarter)?(quarter.startsWith('Q')?quarter:`Q${quarter}`):quarter}`;
    if(year)return year;
    const raw=String(d.rd||d.releaseDate||'').trim();
    const match=/^(20\d{2})[-/]([01]?\d)/.exec(raw);
    if(match){const month=Number(match[2]);return `${match[1]}/Q${Math.ceil(month/3)}`}
    return '';
  }
  function launchPriceInfo(d){
    const raw=d.lp??d.launchPrice??'';if(raw===null||raw===undefined||raw==='')return {available:false,value:'',label:'发布价格',note:'原厂首发价格未核验'};
    const status=String(d.lps||d.launchPriceStatus||'').toLowerCase();
    if(status&& !/(official|manufacturer|launch|verified|datasheet|selector)/.test(status))return {available:false,value:'',label:'发布价格',note:'原厂首发价格未核验'};
    const price=Number(raw);if(!Number.isFinite(price)||price<=0)return {available:false,value:'',label:'发布价格',note:'原厂首发价格未核验'};
    const currency=String(d.lc||d.launchPriceCurrency||'USD').toUpperCase();
    const launch=/(launch|release|initial|intro)/.test(status);
    const formatted=price.toFixed(4).replace(/\.?(0+)$/,'');
    return {available:true,value:`${currency} ${formatted}`,label:launch?'发布价格':'官方参考价',note:launch?'来源明确标注为首发价':'官方产品选择器挂牌价，未标注为首发价'};
  }
  function launchPriceText(d){return launchPriceInfo(d).value||'未核验'}
  function releaseSourceText(source,fallback){
    const value=String(source||'').trim();
    if(!value)return fallback;
    if(/products\.espressif\.com/i.test(value))return '乐鑫官方产品选择器';
    if(/^https?:\/\//i.test(value))return '原厂官方来源';
    return value;
  }
  function releaseFacts(d){
    const dateSource=d.rds||d.releaseDateSource||'',price=launchPriceInfo(d),priceSource=d.lpsrc||d.launchPriceSource||'';
    if(!price.available)return '';
    const date=releaseDateText(d);
    const dateBlock=date?`<div class="release-fact"><span>发布时间</span><b>${esc(date)}</b><small>${esc(releaseSourceText(dateSource,'原厂首发资料'))}</small></div>`:'';
    const priceBlock=`<div class="release-fact"><span>${esc(price.label)}</span><b>${esc(price.value)}</b><small>${esc(priceSource?`${releaseSourceText(priceSource,'原厂官方来源')} · ${price.note}`:price.note)}</small></div>`;
    return `<div class="release-facts">${dateBlock}${priceBlock}</div>`;
  }
  function quoteEndpoint(){
    const configured=String(window.MCUS_QUOTE_API||'').trim();
    if(configured)return configured;
    if(location.protocol==='https:'||location.protocol==='http:')return new URL('/api/quotes',location.href).href;
    return '';
  }
  function safeHttpUrl(value){try{const url=new URL(value);return url.protocol==='https:'||url.protocol==='http:'?url.href:''}catch(_){return ''}}
  function safeHttpUrls(value){return [...new Set(String(value||'').split(/[;\r\n]+/).map(item=>safeHttpUrl(item.trim())).filter(Boolean))]}
  function quoteMessage(code,fallback){
    const messages={feature_disabled:'实时询价暂未开放。',not_configured:'云汉芯城询价尚未配置。',invalid_part:'该订货号不适合直接询价。',invalid_quantity:'询价数量无效。',ickey_auth_error:'云汉芯城鉴权失败，请检查开放平台配置。',ickey_invalid_config:'云汉芯城接口配置无效。',ickey_api_error:'云汉芯城接口暂时不可用，请稍后重试。',no_strict_matches:'没有找到与完整订货号精确匹配的可售货源。',network_error:'无法连接询价服务，请检查网络后重试。'};
    return messages[code]||fallback||'询价失败，请稍后重试。';
  }
  function quotePanelHtml(part,data){
    const quotes=Array.isArray(data.quotes)?data.quotes:[];
    const quantity=Math.max(1,Number(data.quantity)||1);
    const updated=data.updatedAt?new Date(data.updatedAt).toLocaleString('zh-CN',{hour12:false}):'刚刚';
    return `<div class="quote-head"><div><b>云汉芯城询价 · ${esc(part)}</b><br><span>${quotes.length} 条精确货源 · 按 ${quantity} 件 · ${esc(updated)}</span></div></div><form class="quote-quantity" data-quote-quantity><label>询价数量</label><input name="quantity" type="number" min="1" max="1000000" step="1" value="${quantity}" inputmode="numeric"><button type="submit">更新</button></form>${quotes.length?`<div class="quote-list">${quotes.map(item=>{const link=safeHttpUrl(item.url);const price=Number(item.price);const stock=Number(item.stock);const moq=Number(item.moq);const tiers=Array.isArray(item.priceTiers)?item.priceTiers:[];return `<div class="quote-row"><div><span class="quote-shop">${esc(item.shop||'云汉芯城')}</span><p class="quote-title">${esc(item.title||part)}</p><p class="quote-meta">库存 ${Number.isFinite(stock)?stock.toLocaleString('zh-CN'):'—'} · MOQ ${Number.isFinite(moq)?moq:'—'}${item.package?` · ${esc(item.package)}`:''}${item.dateCode?` · 批次 ${esc(item.dateCode)}`:''}${item.leadTime?` · ${esc(item.leadTime)}`:''}</p>${tiers.length?`<div class="quote-tiers">${tiers.slice(0,4).map(tier=>`<span>${esc(tier.quantity)}+ ¥${Number(tier.price).toFixed(4)}</span>`).join('')}</div>`:''}</div><div class="quote-price"><b>¥${Number.isFinite(price)?price.toFixed(4):'—'}</b><small>/ 件</small>${link?`<a href="${esc(link)}">查看货源 ↗</a>`:''}</div></div>`}).join('')}</div>`:`<div class="quote-status">${esc(quoteMessage('no_strict_matches'))}</div>`}<p class="quote-note">数据来自云汉芯城开放平台，按完整订货号精确匹配。“云汉在库、云汉优选、国内现货、授权代理”等是货源类型，不保证对应不同商家；库存、批次、交期及最终含税成交价以云汉结算页为准。</p>`;
  }
  async function requestQuotes(part,quantity=1){
    const panel=$('#quote-panel');if(!panel)return;
    part=String(part||'').trim().toUpperCase();
    quantity=Math.max(1,Math.min(1000000,Math.floor(Number(quantity)||1)));
    if(!/^[A-Z0-9][A-Z0-9+._\/-]{3,63}$/.test(part)){panel.hidden=false;panel.innerHTML=`<div class="quote-head"><b>实时询价</b></div><div class="quote-status">${esc(quoteMessage('invalid_part'))}</div>`;return}
    document.querySelectorAll('[data-quote-part]').forEach(button=>button.classList.toggle('active',button.dataset.quotePart===part));
    panel.hidden=false;panel.dataset.part=part;panel.dataset.quantity=String(quantity);panel.innerHTML=`<div class="quote-head"><b>云汉芯城询价 · ${esc(part)}</b></div><div class="quote-status">正在查询精确型号的实时库存与阶梯价…</div>`;
    panel.scrollIntoView({behavior:'smooth',block:'nearest'});
    const endpoint=quoteEndpoint();
    if(!endpoint){panel.innerHTML=`<div class="quote-head"><b>云汉芯城询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage('not_configured'))}</div>`;return}
    if(quoteAbort)quoteAbort.abort();quoteAbort=new AbortController();
    try{
      const url=new URL(endpoint,location.href);url.searchParams.set('part',part);url.searchParams.set('quantity',String(quantity));
      const response=await fetch(url.href,{headers:{accept:'application/json'},signal:quoteAbort.signal});
      const data=await response.json().catch(()=>({}));
      if(panel.dataset.part!==part)return;
      if(!response.ok)throw Object.assign(new Error(data.message||''),{code:data.code||'ickey_api_error'});
      panel.innerHTML=quotePanelHtml(part,data);
      const quantityForm=panel.querySelector('[data-quote-quantity]');if(quantityForm)quantityForm.onsubmit=event=>{event.preventDefault();requestQuotes(part,quantityForm.elements.quantity.value)};
    }catch(error){
      if(error&&error.name==='AbortError')return;
      if(panel.dataset.part!==part)return;
      const code=error&&error.code?error.code:'network_error';
      panel.innerHTML=`<div class="quote-head"><b>云汉芯城询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage(code,error&&error.message))}<br><button class="quote-retry" data-quote-retry="${esc(part)}">重新询价</button></div>`;
      const retry=panel.querySelector('[data-quote-retry]');if(retry)retry.onclick=()=>requestQuotes(retry.dataset.quoteRetry,panel.dataset.quantity);
    }
  }
  function openDetail(id){
    state.detail=id;const d=byId.get(id);if(!d)return;
    const capabilityTypes=new Set([
      'VendorCapability','Accelerator','PowerOther','Audio','I3C','USBSS','USBOTG','Camera','LCD','GLCD',
      'Crypto','RNG','NPU','ExtBus','PSRAM','RTC_RAM','Security','CoreOther','MCPWM','LEDPWM','RMT','Touch','Hall','TOF',
      'DMA2D','CORDIC','FMAC','MDF','DFSDM','JPEG','SAI','PDMIC','SPDIFRX','CANFD','FlexRay','UCPD','USBPD','USBHS',
      'MIPI','HDMI_CEC','DMAMUX','DMAChannels','LINKEDLIST','ICACHE','DCACHE','RAMECC','FLEXRAMECC','OTFDEC','GTZC','SAU','IDAU',
      'HSEM','EVSYS','CCL','PDEC','PIO','HSP_Engine','HSP1','DIVAS','VREFBUF','SDADC','SUBGHZ','BLE','ZIGBEE','SWPMI',
      'AES','AESB','PKA','CRCCU','CRCSCAN','BSEC','HSM',
    ]);
    const sourceCapabilities=(d.pi||[]).filter(item=>{const type=String(item?.t||'');const name=String(item?.n||'');return capabilityTypes.has(type)||/^WCH\s|^(?:NeoChrom|Chrom-ART|HSP1|NPU|CORDIC|FMAC|DMA2D|MDF|DFSDM|JPEG|SAI|PDM|CAN\s*FD|UCPD|USB\s*PD)/i.test(name)}).flatMap(capabilityLabels);
    const confirmedLabels=[...(d.acc||[]),...(d.feat||[]),...sourceCapabilities].map(featureLabel).filter(Boolean);const confirmedUnique=[...new Set(confirmedLabels)];const accelerators=confirmedUnique.map(item=>acceleratorEntry(item,'confirmed'));const features=accelerators;const missingReasons=missingInfoReasons(d);const parts=d.parts||[];const selected=state.compare.has(d.id);const layer=$('#detail-layer');const sourceActions=safeHttpUrls(d.src).map((url,index,urls)=>`<a class="detail-action" href="${esc(url)}">${urls.length>1?`打开来源页面 ${index+1}`:'打开来源页面'} ↗</a>`).join('');
    const facts=engineeringFacts(d);
    const coreBody=`<div class="spec-grid">${spec(engineeringValue(d.c||d.a),'处理器核心')}${spec(d.cc?d.cc+' core':'未核验','核心数')}${spec(d.hz?clock(d.hz):'未核验','最高核心频率')}${spec(voltageRange(d)==='—'?'未核验':voltageRange(d),'工作电压范围')}${spec(engineeringMemory(d.fl),'片上 Flash')}${spec(engineeringMemory(d.ra),'片上 RAM')}${spec(yesNoValue(d.fpu),'FPU')}${spec(yesNoValue(d.dsp),'DSP')}${spec(yesNoValue(d.mpu),'MPU')}${spec(yesNoValue(d.tz),'TrustZone')}</div><h3 class="detail-subheading">Flash 工程属性</h3><div class="spec-grid">${spec(facts.flashFacts.length?facts.flashFacts.join(' · '):'未核验','等待周期 / Bank / ECC')}${spec(facts.cache?'有':'未核验','Cache')}${spec(Array.isArray(d.eb)?(facts.externalBus.length?facts.externalBus.join(' / '):'无'):'未核验','外部存储总线')}</div><h3 class="detail-subheading">RAM 结构</h3><div class="spec-grid">${spec(facts.ramRegions.length?facts.ramRegions.join(' · '):'未核验','RAM 分区')}${spec(facts.ramTypes.length?facts.ramTypes.join(' / '):'未核验','RAM 类型')}${spec(facts.ramEcc?'有':'未核验','RAM ECC')}${spec(facts.multiCore?(facts.exclusiveRam?'有':'未核验'):'不适用','多核独占 RAM')}</div>`;
    const peripheralBody=`<div class="spec-grid">${spec(facts.timerWidths.length?facts.timerWidths.join(' / ')+' bit':'未核验','定时器位宽')}${spec(engineeringValue(d.tim),'TIM 总数')}${spec(engineeringValue(d.adcu),'ADC 转换器单元')}${spec(engineeringValue(d.adch),'ADC 通道（不含引脚）')}${spec(facts.adcResolution?facts.adcResolution+' bit':'未核验','ADC 分辨率')}${spec(facts.adcRate?clock(facts.adcRate):'未核验','ADC 采样率')}${spec(engineeringValue(d.dac),'DAC 单元')}${spec(facts.dacResolution?facts.dacResolution+' bit':'未核验','DAC 分辨率')}${spec(facts.dacRate?clock(facts.dacRate):'未核验','DAC 速度')}${spec(engineeringValue(d.gpio),'GPIO')}${spec(engineeringValue(d.dma),'DMA 通道')}${spec(engineeringValue(d.sercom),'SERCOM / FLEXCOM')}${spec(engineeringValue(d.spi),'SPI')}${spec(engineeringValue(d.i2c),'I²C')}${spec(engineeringValue(d.usart),'USART')}${spec(engineeringValue(d.uart),'UART')}${spec(engineeringValue(d.can),'CAN')}${spec(engineeringValue(d.usbd),'USB Device')}${spec(engineeringValue(d.usbh),'USB Host')}${spec(engineeringValue(d.eth),'Ethernet')}${spec(engineeringValue(d.pin),'封装引脚数')}</div>`;
    const featureBody=`<div class="feature-panel">${accelerators.length?`<div class="feature-label">已确认的加速器与厂商能力 · 点击查看说明</div>${acceleratorGrid(accelerators)}`:'<div class="feature-label">当前来源没有已确认的专用加速器或能力记录</div>'}${(d.pending||[]).length?`<div class="feature-label feature-label-spaced">待逐器件官方文档核验</div><div class="feature-list">${d.pending.map(x=>{const label=featureLabel(x);return `<span class="feature-chip pending" title="${esc(label)}">${esc(label)}</span>`}).join('')}</div><p class="feature-note">候选项不代表该具体后缀型号已经确认支持。</p>`:''}</div>`;
    const scoreBody=`<div class="score-panel">${[['计算',d.cs],['存储',d.ms],['外设',d.ps],['加速器',d.acs]].map(x=>`<div class="score-row"><label>${x[0]}</label><div class="score-bar"><i style="width:${Math.max(0,Math.min(100,x[1]||0))}%"></i></div><b>${value(x[1])}</b></div>`).join('')}<p class="score-note">数据覆盖率 ${value(d.cov)}%。这是选型排序指标，不是实测性能。CoreMark：${value(d.cm)} · DMIPS：${value(d.dm)}。</p></div>`;
    const partsBody=parts.length?`<div class="parts">${parts.map(p=>`<div class="part-row"><div><b>${esc(p.n)}</b><p>后缀 ${esc(p.s||'—')} · 封装码 ${esc(p.p||'—')} · 温度码 ${esc(p.t||'—')} · 包装 ${esc(p.k||'—')}</p></div><div class="part-actions"><span class="verified">✓ 已核验</span><button class="quote-trigger" data-quote-part="${esc(p.n)}">询价</button></div></div>`).join('')}</div>`:'<div class="feature-panel"><div class="feature-label">完整订货号尚未导入；不会根据后缀组合自动生成。请明确输入厂商完整订货号后询价。</div><form class="quote-manual" id="quote-manual"><input id="quote-manual-part" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="输入完整订货号，如 STM32F429ZIT6"><button type="submit">询价</button></form></div>';
    const warningTone=missingInfoLevel(missingReasons)==='manual'?' manual-warning':'';
    layer.innerHTML=`<div class="detail-backdrop"></div><section class="detail-page"><div class="detail-header ${missingInfoHeaderClass(missingReasons)}"><button class="back-btn">‹</button><div class="detail-title"><h1>${missingInfoBadge(d)}${esc(d.n)}</h1><p>${esc(d.m)} · ${esc(d.s)} · ${esc(productType(d.pt))}</p></div><div class="detail-score"><b>${value(d.idx)}</b><span>选型指数 / 100</span></div></div><div class="detail-hero"><div class="detail-path">${esc(d.m)} › ${esc(d.s)} › ${esc(d.l)} › ${esc(productType(d.pt))}</div><div class="detail-model"><b>${esc(d.n)}</b><span>变体码 ${esc(d.v||'—')}</span></div>${missingReasons.length?`<div class="data-warning-note${warningTone}">⚠ 关键资料缺失：${esc(missingReasons.join('、'))}</div>`:''}${releaseFacts(d)}</div>${detailAccordion('核心与存储',coreBody,true,d.c||d.a||'核心')} ${powerSection(d)}${detailAccordion('外设资源',peripheralBody,true,`${[d.tim,d.adcu,d.adch,d.gpio,d.uart,d.usart].filter(v=>v!==undefined&&v!==null&&v!=='').length} 类已知`)}${packageSection(d)}${documentsSection(d)}${inventorySection(d)}${detailAccordion('厂商加速器与特性',featureBody,false,features.length?`${features.length} 项`:'未确认')}${detailAccordion('评分拆解',scoreBody,false,d.idx?`${d.idx} / 100`:'未评分')}${detailAccordion('官方完整订货号',`${partsBody}<div class="quote-panel" id="quote-panel" hidden aria-live="polite"></div>`,false,`${parts.length} 个`)}<button class="detail-action primary" id="detail-compare">${selected?'✓ 已加入对比':'＋ 加入参数对比'}</button>${sourceActions}</section>`;
    if(!quotesEnabled){const manual=layer.querySelector('.quote-manual');if(manual){const label=manual.parentElement.querySelector('.feature-label');if(label)label.textContent='完整订货号尚未导入；不会根据后缀组合自动生成。'}layer.querySelectorAll('.quote-trigger,.quote-manual,.quote-panel').forEach(element=>element.remove())}
    const detailSpecGrids=layer.querySelectorAll('.spec-grid');
    const coreCountValue=detailSpecGrids[0]?.querySelector('.spec-cell:nth-child(2) b');
    if(coreCountValue)coreCountValue.textContent=d.cc?d.cc+' core':'—';
    if(detailSpecGrids[1])detailSpecGrids[1].insertAdjacentHTML('beforeend',spec(value(d.usb),'USB（角色未标明）'));
    if(d.m==='Microchip'){
      layer.querySelector('.detail-title p').textContent=`${vendorName(d.m)} · ${d.s}`;
      const path=layer.querySelector('.detail-path');path.textContent=path.textContent.replace(/^Microchip/,vendorName(d.m));
    }
    if((d.boards||[]).length)layer.querySelector('.detail-hero').insertAdjacentHTML('beforeend',boardTags(d,true));
    requestAnimationFrame(()=>layer.classList.add('open'));layer.querySelector('.back-btn').onclick=closeDetail;layer.querySelector('.detail-backdrop').onclick=closeDetail;$('#detail-compare').onclick=()=>{toggleCompare(d.id);openDetail(d.id)};layer.querySelectorAll('[data-accelerator-toggle]').forEach(button=>button.onclick=()=>{const card=button.closest('[data-accelerator-card]'),expanded=card.classList.toggle('active');button.setAttribute('aria-expanded',String(expanded));const explanation=card.querySelector('[data-accelerator-explanation]');if(explanation)explanation.hidden=!expanded;if(expanded)layer.querySelectorAll('[data-accelerator-card].active').forEach(other=>{if(other===card)return;other.classList.remove('active');const toggle=other.querySelector('[data-accelerator-toggle]');const body=other.querySelector('[data-accelerator-explanation]');if(toggle)toggle.setAttribute('aria-expanded','false');if(body)body.hidden=true})});layer.querySelectorAll('[data-quote-part]').forEach(button=>button.onclick=()=>requestQuotes(button.dataset.quotePart));const manualForm=layer.querySelector('#quote-manual');if(manualForm)manualForm.onsubmit=event=>{event.preventDefault();requestQuotes(layer.querySelector('#quote-manual-part').value)};
  }
  function closeDetail(){state.detail=null;if(quoteAbort){quoteAbort.abort();quoteAbort=null}const layer=$('#detail-layer');layer.classList.remove('open');setTimeout(()=>{if(!state.detail)layer.innerHTML=''},220)}
  function render(){renderHeader();if(state.tab==='catalog')renderCatalog();else if(state.tab==='search')renderSearch(false);else if(state.tab==='assistant')renderAssistant();else if(state.tab==='compare')renderCompare();else if(state.tab==='data')renderData();else renderAuthor();updateNav()}
  window.MCUL={handleAndroidBack:function(){if(state.detail){closeDetail();return true}if(state.tab==='catalog'){if(state.browse.line){state.browse.line=null;renderCatalog();return true}if(state.browse.series){state.browse.series=null;renderCatalog();return true}if(state.browse.vendor){state.browse.vendor=null;renderCatalog();return true}}if(state.tab!=='catalog'){setTab('catalog');return true}return false}};
  document.querySelectorAll('#bottom-nav button').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
  state.assistantMessages=restoreAssistant();
  $('#snapshot-pill').textContent=`${count(catalog.meta.devices)} DEVICES · OFFLINE`;
  $('#splash').remove();$('#app').hidden=false;render();if(previewDevice&&byId.has(previewDevice))setTimeout(()=>openDetail(previewDevice),50);
  installKeyboardViewport();
})();
