(function(){
  'use strict';
  const catalog=window.MCU_CATALOG;
  if(!catalog){document.querySelector('.splash-sub').textContent='目录载入失败';return;}
  const $=s=>document.querySelector(s);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const value=v=>v===undefined||v===null||v===''?'—':v;
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
  devices.forEach(d=>{const peripheralText=(d.pi||[]).flatMap(item=>[item.n,item.t,item.d]).join(' ');const aliases=peripheralFilters.filter(item=>peripheralCount(d,item.key)).map(item=>item.aliases).join(' ');const vendorAliases=d.m==='Qinheng'?'沁恒 wch qinheng nanjing qinheng microelectronics qingke 青稞':d.m==='STC'?'stc 宏晶 hongjing stc microelectronics 8051':d.m==='HPMicro'?'先楫 hpm hpmicro risc-v 上海先楫':d.m==='Renesas'?'瑞萨 renesas ra rx rl78 rh850 synergy risc-v 瑞萨电子':d.m==='Allwinner'?'全志 allwinner xradio 芯之联 wireless mcu 实时 异构 soc 数传':d.m==='MicroPy MCU'?'micropy micropython mpy raspberry pi rp2040 rp2350 kendryte k210 micropython mcu':' ';d._q=[d.n,d.l,d.s,d.f,d.m,d.pt,vendorAliases,d.v,d.c,d.a,peripheralText,aliases,...(d.boards||[]),...(d.parts||[]).map(p=>p.n)].join(' ').toLowerCase()});
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
  function vendorName(name){if(name==='Allwinner')return '全志（Allwinner / XRadio）';if(name==='Artery')return '雅特力（Artery）';if(name==='Microchip')return 'Microchip（原 Atmel）';if(name==='Qinheng')return '沁恒（WCH）';if(name==='Renesas')return '瑞萨电子（Renesas）';if(name==='STC')return 'STC（宏晶）';if(name==='HPMicro')return '先楫半导体（HPMicro）';return name}
  function productType(type){return ({wireless_mcu:'无线 MCU',wireless_audio_mcu_soc:'无线音频 MCU SoC',wireless_connectivity_chip:'无线连接芯片',heterogeneous_realtime_soc:'带实时 MCU 核的 SoC',micropython_mcu:'MicroPython MCU'})[type]||'MCU'}
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
    $('#view').innerHTML=`${breadcrumb([{level:'root',label:'芯片目录'},{level:'vendor',label:browse.vendor},{level:'series',label:browse.series},{level:'line',label:browse.line}])}<div class="page-heading"><h1>${esc(browse.line)}</h1><p>器件变体层保留厂商标注的封装、引脚、存储和版本后缀，不把不同变体合并成一个型号。</p></div><div class="summary-strip">${summary('器件变体',count(lineDevices.length))}${summary('完整订货号',count(partCount(lineDevices)))}${summary('最高主频',clock(maxClock(lineDevices)))}</div><div class="section-heading"><h2>具体器件变体</h2><span>第 3 层</span></div><div class="variant-list">${lineDevices.map(d=>`<button class="variant-row" data-device="${esc(d.id)}"><span class="variant-id"><h3>${esc(d.n)}</h3><p>变体码 ${esc(d.v||'—')} · ${esc(d.c||d.a||'—')} · ${clock(d.hz)} · ${d.sercom!==undefined?'SERCOM/FLEXCOM '+value(d.sercom):'UART '+value(d.uart)} · ${memory(d.fl)} Flash · ${memory(d.ra)} RAM</p></span><span class="variant-side"><b>${value(d.idx)}</b><span>选型指数 ${(d.parts||[]).length?`<i class="variant-parts">${(d.parts||[]).length} 订货号</i>`:''}</span></span></button>`).join('')}</div>`;
    bindCrumbs();document.querySelectorAll('[data-device]').forEach(b=>b.onclick=()=>openDetail(b.dataset.device));
  }
  function bindCrumbs(){document.querySelectorAll('[data-crumb]').forEach(b=>b.onclick=()=>{const level=b.dataset.crumb;if(level==='root'){state.browse.vendor=null;state.browse.series=null;state.browse.line=null}else if(level==='vendor'){state.browse.series=null;state.browse.line=null}else if(level==='series'){state.browse.line=null}$('#view').scrollTop=0;renderCatalog()})}
  function filtered(){const q=state.query.trim().toLowerCase();const list=devices.filter(d=>(!q||d._q.includes(q))&&(!state.vendorFilter||d.m===state.vendorFilter)&&(!state.coreFilter||(d.c||d.a)===state.coreFilter)&&(!state.peripheralFilter||(peripheralCount(d,state.peripheralFilter)||0)>=state.peripheralMin));list.sort(state.sort==='name'?(a,b)=>natural(a.n,b.n):(a,b)=>(b.idx??-1)-(a.idx??-1)||natural(a.n,b.n));return list}
  function deviceRow(d){const selected=state.compare.has(d.id);const selectedPeripheral=peripheralByKey.get(state.peripheralFilter);const peripheralSpec=selectedPeripheral?`<span>${value(peripheralCount(d,selectedPeripheral.key))}<small>${esc(selectedPeripheral.label)}</small></span>`:`<span>${value(d.sercom!==undefined?d.sercom:(d.usart!==undefined?d.usart:d.uart))}<small>${d.sercom!==undefined?'SERCOM/FLEXCOM':d.usart!==undefined?'USART':'UART'}</small></span>`;return `<button class="device-row" data-device="${esc(d.id)}"><span><span class="device-title"><h3>${esc(d.n)}</h3>${(d.parts||[]).length?`<i>${(d.parts||[]).length} 订货号</i>`:''}</span><p class="device-path">${esc(vendorName(d.m))} › ${esc(d.s)} › ${esc(d.l)}</p>${boardTags(d)}<span class="device-specs"><span>${esc(d.c||d.a||'—')}<small>核心</small></span><span>${clock(d.hz)}<small>主频</small></span><span>${memory(d.fl)}<small>Flash</small></span>${peripheralSpec}</span></span><span class="device-score"><b>${value(d.idx)}</b><span>选型指数</span><i class="compare-toggle ${selected?'selected':''}" data-compare="${esc(d.id)}">${selected?'已对比':'＋ 对比'}</i></span></button>`}
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
      const negBefore=/(?:不要|不需要|无需|不含|排除|禁止|不考虑|不带|没有|不用|不想要|别|不配|不想配)[^，,;。]{0,14}$/i.test(before);
      const negAfter=/^[^，,;。]{0,5}(?:不要|不需要|无需|不含|排除|禁止|不考虑|不带|没有|不用|不想要|别|不配|不想配)/i.test(after);
      if(!negBefore&&!negAfter)continue;
      const soft=/(?:最好|尽量|尽可能|优先|倾向|建议|可以不|能不)[^，,;。]{0,18}$/i.test(before);
      const optional=/(?:也行|也可以|无所谓|没关系|可有可无|不强求|有就行|没有就行)/i.test(before+after);
      return {hard:!soft&&!optional,soft:soft&&!optional,optional};
    }
    return {hard:false,soft:false,optional:false};
  }
  function aiHasNegation(text,terms){return (terms||[]).some(term=>aiRelation(text,String(term||'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).hard)}
  function aiHasSoftNegation(text,terms){return (terms||[]).some(term=>aiRelation(text,String(term||'').replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).soft)}
  function aiHasOptional(text,pattern){return aiRelation(text,pattern).optional}
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
      [/跑得快|跑得动|速度快一点|快一点|处理得快|反应快|响应要快|高频一点|频率高一点|性能别太差/g,' 高性能 '],
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
    const a=String(aliases),source=String(text||''),number=aiQuantityPattern(),unit='(?:个|路|组|颗|项|通道|核)?',minWord='(?:至少|不少于|不低于|大于等于|起码|起步)',maxWord='(?:最多|不超过|不高于|小于等于|至多)';
    const minBefore=new RegExp(minWord+'?\\s*'+number+'\\s*'+unit+'\\s*(?:以上|及以上)?\\s*(?:'+a+')','i').exec(source);
    const minAfter=new RegExp('(?:'+a+')\\s*'+minWord+'?\\s*'+number+'\\s*'+unit+'\\s*(?:以上|及以上)?','i').exec(source);
    const explicitMinBefore=new RegExp(minWord+'\\s*'+number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const explicitMinAfter=new RegExp('(?:'+a+')\\s*'+minWord+'\\s*'+number,'i').exec(source);
    const maxBefore=new RegExp(maxWord+'\\s*'+number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const maxAfter=new RegExp('(?:'+a+')[^，,;。]{0,12}'+maxWord+'\\s*'+number,'i').exec(source);
    const plainBefore=new RegExp(number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source);
    const plainAfter=new RegExp('(?:'+a+')\\s*'+number,'i').exec(source);
    const pick=(hit,mode)=>{if(!hit||hit[1]===undefined)return null;const range=aiQuantityRange(hit[1]);return range?(mode==='max'?range.max:mode==='target'?range.target:range.min):aiNumberValue(hit[1])};
    const min=pick(explicitMinBefore)||pick(explicitMinAfter)||(!new RegExp(maxWord,'i').test(source)?(pick(minBefore)||pick(minAfter)||pick(plainBefore)||pick(plainAfter)||null):null);
    const max=pick(maxBefore,'max')||pick(maxAfter,'max')||null;
    const targetHit=new RegExp(number+'\\s*'+unit+'\\s*(?:'+a+')','i').exec(source)||new RegExp('(?:'+a+')[^，,;。]{0,8}'+number+'\\s*'+unit,'i').exec(source);
    const targetHitRange=targetHit&&aiQuantityRange(targetHit[1]);
    const target=(targetHit&&(targetHitRange||aiNearConstraint(source,a)))?pick(targetHit,'target'):null;
    return {min:max?null:min,max,target,approx:Boolean(target)};
  }
  function aiMemory(text,aliases){const number=aiQuantityPattern(),unit='(gb|mb|kb|g|m|k|吉|兆)';const after=new RegExp('(?:'+aliases+')[^\\d零〇一二两三四五六七八九十百千万]{0,14}'+number+'\\s*'+unit,'i').exec(text);const before=new RegExp(number+'\\s*'+unit+'\\s*(?:'+aliases+')','i').exec(text);const m=after||before;return m?aiUnitBytes(m[1],m[2]==='吉'?'gb':m[2]==='兆'?'mb':m[2]):null}
  function aiCoreFromText(text){
    const source=String(text||'').toLowerCase();
    // Longest-first plus an alphanumeric boundary prevents m33 from being cut to m3.
    const pattern=/(?:^|[^a-z0-9])((?:cortex[- ]?m(?:35p|55|33|23|0\+?|7|4|3)|arm\s+(?:cortex[- ]?)?m(?:35p|55|33|23|0\+?|7|4|3)|m(?:35p|55|33|23|0\+?|7|4|3)(?:f)?))(?=$|[^a-z0-9])/i;
    const hit=pattern.exec(source)||/(?:^|[^a-z0-9])((?:cortex[- ]?m|arm\s+(?:cortex[- ]?)?m|m)\d+\+?(?:f)?)(?=$|[^a-z0-9])/i.exec(source);if(!hit)return null;
    const token=hit[1].replace(/\s+/g,'').replace(/^arm/i,'').replace(/^cortex-/i,'').replace(/^cortex/i,'');
    const match=/m(?:35p|55|33|23|0\+?|7|4|3)/i.exec(token)||/m\d+\+?/i.exec(token);return match?'cortex-'+match[0].toLowerCase():null;
  }
  function aiMetric(d,key){if(key==='serial'){const values=[d.uart,d.usart].filter(v=>typeof v==='number');return values.length?values.reduce((a,b)=>a+b,0):null}if(key==='usbAny'){const values=[d.usb,d.usbd,d.usbh,d.otg].filter(v=>typeof v==='number');return values.length?Math.max(...values):null}return peripheralCount(d,key)}
  function aiKnownWirelessAbsence(d,key){
    if(key!=='wifi'&&key!=='bluetooth')return false;
    // 对通用、低功耗和高性能 MCU，目录没有无线条目即表示未集成该无线能力；
    // 无线 MCU / SoC / 模组则必须保留“未核验”，避免把 Bluetooth 或 Wi-Fi 猜成不存在。
    const ordinary=['general_purpose_mcu','low_power_mcu','high_performance_mcu','micropython_mcu'];
    return ordinary.includes(d.pt)&&Array.isArray(d.pi)&&Number(d.cov)>=90;
  }
  function aiParse(prompt){
    const raw=String(prompt||'').trim(),text=aiNormalizeText(raw);
    const req={prompt:raw,normalized:text,vendor:null,excludedVendors:[],core:null,exact:null,clock:null,clockTarget:null,ram:null,ramTarget:null,flash:null,flashTarget:null,pins:null,pinsMin:null,pinsMax:null,coreCount:null,coreOnly:false,fpu:false,micropython:false,minimums:{},maximums:{},softMinimums:{},softTargets:{},excludedFeatures:[],softExcludedFeatures:[],vagueFeatures:[],features:[],profiles:[],profileLabels:[],preferences:[],warnings:[]};
    const modelIntent=aiModelInfer(text),addUnique=(list,value)=>{if(value&&!list.includes(value))list.push(value)};
    modelIntent.profiles.forEach(profile=>{addUnique(req.profiles,profile.key);addUnique(req.profileLabels,profile.label);Object.entries(profile.softMinimums||{}).forEach(([key,min])=>{if(!(key in req.minimums))req.softMinimums[key]=Math.max(req.softMinimums[key]||0,min)});(profile.preferences||[]).forEach(key=>addUnique(req.preferences,key))});
    modelIntent.preferences.forEach(preference=>addUnique(req.preferences,preference.key));
    const vendors=[
      ['STMicroelectronics',/stm32|stmicroelectronics|意法|st芯片/],['Espressif',/esp32|esp8266|乐鑫|espressif/],['Qinheng',/ch32|沁恒|wch|qingke|青稞/],['HPMicro',/hpmicro|hpm|先楫/],['Microchip',/microchip|atmel|avr|samd|pic/],['STC',/stc|宏晶/],['GigaDevice',/兆易创新|gigadevice|gd32/],['MindMotion',/灵动微|mindmotion|mm32/],['Nuvoton',/新唐|nuvoton|numicro/],['Puya',/普冉|puya|py32/],['Geehy',/极海|geehy|apm32/],['Infineon',/英飞凌|infineon|psoc|xmc/],['Texas Instruments',/德州仪器|ti芯片|texas instruments|mspm|msp430/],['Renesas',/瑞萨|renesas|\bra[02468][a-z0-9-]*\b|\brx[0-9][a-z0-9-]*\b|rl78|rh850|synergy/],['Allwinner',/全志|allwinner|xradio|xr806/],['MicroPy MCU',/micropython|micropy|canmv|rp2040|rp2350|rp2354|k210|k230|k510|树莓派|kendryte|嘉楠/]
    ];
    localModel.vendors.forEach(item=>{if(aiHasNegation(text,item.terms))req.excludedVendors.push(item.label)});
    const vendor=vendors.find(item=>item[1].test(text));if(modelIntent.vendor&&!req.excludedVendors.includes(modelIntent.vendor))req.vendor=modelIntent.vendor;else if(vendor&&!req.excludedVendors.includes(vendor[0]))req.vendor=vendor[0];
    const exact=/\b(?:stm32|esp32|esp8266|ch32|gd32|mm32|py32|apm32|rp2040|rp2350[a-z0-9]*|rp2354[a-z0-9]*|k210|k230d?|k510|hpm[0-9a-z]+|ra[02468][a-z0-9-]*|rx[0-9][a-z0-9-]*|r7[a-z0-9-]+|r5f[a-z0-9-]+|r9a[a-z0-9-]+)[a-z0-9-]*\b/i.exec(text);if(exact)req.exact=exact[0].toLowerCase();
    const coreAliases=[...cores].sort((a,b)=>b.length-a.length),core=coreAliases.find(item=>text.includes(String(item).toLowerCase())),explicitCore=aiCoreFromText(text);if(explicitCore)req.core=explicitCore;else if(modelIntent.core)req.core=modelIntent.core;else if(core)req.core=core;
    const quantity=aiQuantityPattern(),frequencyUnit='(ghz|mhz|khz|兆|m(?!b)|g(?!b)|k(?!b))',clockPattern=new RegExp('(?:主频|频率|时钟|最高|clock|速度)[^\\d零〇一二两三四五六七八九十百千万]{0,14}'+quantity+'\\s*'+frequencyUnit+'?','i'),clockFallback=new RegExp(quantity+'\\s*'+frequencyUnit,'i'),clockMatch=clockPattern.exec(text)||clockFallback.exec(text);if(clockMatch){const unit=clockMatch[2]==='兆'?'mhz':clockMatch[2]||'mhz',clockValue=aiFrequency(clockMatch[1],unit),clockContext=text.slice(Math.max(0,text.indexOf(clockMatch[0])-8),text.indexOf(clockMatch[0])+clockMatch[0].length+8);if(clockValue){if(/约|大约|左右|上下|接近|差不多|附近/.test(clockContext))req.clockTarget=clockValue;else req.clock=clockValue}}
    const performanceRelation=aiRelation(text,'高频|高速|高性能|主频');
    if(/(?:主频|频率|速度)[^，,;。]{0,12}(?:越高|越快|高一些|高速)|高频|高速|高性能/i.test(text)&&!performanceRelation.hard&&!performanceRelation.soft)addUnique(req.preferences,'highPerformance');
    const ramValue=aiMemory(text,'ram|sram|内存'),flashValue=aiMemory(text,'flash|闪存');if(ramValue){if(aiNearConstraint(text,'ram|sram|内存'))req.ramTarget=ramValue;else req.ram=ramValue}if(flashValue){if(aiNearConstraint(text,'flash|闪存'))req.flashTarget=flashValue;else req.flash=flashValue}
    const coreCountMatch=new RegExp('(单|双|三|四|五|六|七|八|\\d+)\\s*(?:核|核心)','i').exec(text);if(coreCountMatch){const map={单:1,双:2,三:3,四:4,五:5,六:6,七:7,八:8};req.coreCount=map[coreCountMatch[1]]||Number(coreCountMatch[1]);req.coreOnly=/单核|单核心|纯\s*核/i.test(text)}
    const pinAfter=new RegExp(quantity+'\\s*(?:pin|脚|引脚)','i').exec(text),pinBefore=new RegExp('(?:引脚|pin|脚)[^\\d零〇一二两三四五六七八九十百千万]{0,8}'+quantity,'i').exec(text),pinMatch=pinAfter||pinBefore;if(pinMatch){const pinValue=aiNumberValue(pinMatch[1]||pinMatch[2]),pinContext=text.slice(Math.max(0,text.indexOf(pinMatch[0])-8),text.indexOf(pinMatch[0])+pinMatch[0].length+8);if(/以内|以下|不超过|最多|小于等于/.test(pinContext))req.pinsMax=pinValue;else if(/至少|不少于|不低于|以上|大于等于/.test(pinContext))req.pinsMin=pinValue;else req.pins=pinValue}
    req.fpu=/\bfpu\b|浮点|硬件浮点/i.test(text);req.micropython=/micropython|micropy|canmv|micro\s*python/i.test(text);
    const peripheralAliases=[
      ['serial','\\buart\\b|\\busart\\b|串口|串行|通信口'],['spi','\\bspi\\b|串行外设'],['i2c','\\bi2c\\b|i²c|两线总线'],['i2s','\\bi2s\\b|i²s|音频接口'],['can','\\bcan(?:fd)?\\b|\\btwai\\b|can总线'],['usbh','usb\\s*host|usb主机|host usb'],['usbd','usb\\s*device|usb设备|device usb'],['usbAny','\\busb\\b|usb设备|usb主机|otg'],['eth','ethernet|以太网'],['wifi','wi[- ]?fi|wifi|无线局域网|无线'],['bluetooth','bluetooth|蓝牙|\\bble\\b'],['cam','camera|摄像头|相机|dvp|dcmi'],['display','display|lcd|显示'],['pwm','\\bpwm\\b|脉宽|电机'],['adch','adc|模拟通道'],['gpio','\\bgpio\\b|通用io'],['tim','timer|定时器|计数器']
    ];
    peripheralAliases.forEach(([key,aliases])=>{
      const mentioned=new RegExp(aliases,'i').test(text),relation=aiRelation(text,aliases),optional=aiHasOptional(text,aliases),softMention=aiHasSoftQualifier(text,aliases),constraint=aiConstraint(text,aliases);
      if(relation.hard||optional){delete req.minimums[key];delete req.maximums[key];delete req.softTargets[key];if(relation.hard)addUnique(req.excludedFeatures,key);if(optional)addUnique(req.vagueFeatures,key);return}
      if(relation.soft){delete req.minimums[key];delete req.maximums[key];delete req.softTargets[key];addUnique(req.softExcludedFeatures,key);return}
      if(softMention){delete req.minimums[key];delete req.maximums[key];if(constraint.target!==null)req.softTargets[key]=constraint.target;if(constraint.min!==null)req.softMinimums[key]=Math.max(req.softMinimums[key]||0,constraint.min);else if(mentioned)req.softMinimums[key]=Math.max(req.softMinimums[key]||0,1);return}
      if(constraint.min!==null)req.minimums[key]=constraint.min;if(constraint.max!==null)req.maximums[key]=constraint.max;if(constraint.target!==null)req.softTargets[key]=constraint.target;
      if(constraint.min===null&&constraint.max===null&&!constraint.target&&mentioned){if(/够用|有就行|有即可|不用太多|随便几路|几路|若干|一些/i.test(text))addUnique(req.vagueFeatures,key);else req.minimums[key]=1}
    });
    if(req.minimums.usbh||req.minimums.usbd||req.maximums.usbh||req.maximums.usbd){delete req.minimums.usbAny;delete req.maximums.usbAny}
    modelIntent.features.forEach(key=>{const modelFeature=localModel.features.find(item=>item.key===key),pattern=(modelFeature?.terms||[]).map(term=>String(term).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),relation=pattern?aiRelation(text,pattern):{hard:false,soft:false,optional:false};if(relation.hard){addUnique(req.excludedFeatures,key);delete req.minimums[key];delete req.maximums[key]}else if(relation.soft){addUnique(req.softExcludedFeatures,key);delete req.minimums[key];delete req.maximums[key]}else if(key==='usbAny'&&(req.minimums.usbh||req.minimums.usbd)){}else if(req.vagueFeatures.includes(key)){}else if(!(key in req.minimums)&&!(key in req.softMinimums))req.minimums[key]=1});
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
    if(req.minimums.wifi)req.features.push('Wi-Fi');if(req.minimums.bluetooth)req.features.push('Bluetooth');if(req.minimums.cam)req.features.push('摄像头接口');if(req.minimums.display)req.features.push('显示接口');if(req.minimums.can)req.features.push('CAN');if(req.minimums.usbAny)req.features.push('USB');
    return req;
  }
  let aiEvaluationCacheRequest=null,aiEvaluationCache=new Map();
  function aiEvaluate(d,req){
    if(aiEvaluationCacheRequest!==req){aiEvaluationCacheRequest=req;aiEvaluationCache=new Map()}
    const cached=aiEvaluationCache.get(d.id);if(cached)return cached;
    const failures=[],unknowns=[],matched=[];const coreText=String(d.c||d.a||'').toLowerCase();
    const check=(condition,label,unknown)=>{if(unknown)unknowns.push(label);else if(condition)matched.push(label);else failures.push(label)};
    req.excludedVendors.forEach(vendor=>{if(d.m===vendor)failures.push('排除厂商 '+vendor)});
    if(req.vendor)check(d.m===req.vendor,'厂商 '+vendorName(req.vendor),!d.m);
    if(req.core)check(coreText.includes(String(req.core).toLowerCase()),'核心 '+req.core,!coreText);
    if(req.coreOnly)check(d.cc===1&&!/[+/,]/.test(coreText),'单核',typeof d.cc!=='number');
    if(req.coreCount)check(d.cc===req.coreCount,req.coreCount+' 核',typeof d.cc!=='number');
    if(req.micropython)check(d.pt==='micropython_mcu','MicroPython',!d.pt);
    if(req.exact)check(String(d._q||'').includes(req.exact),'型号 '+req.exact,!d._q);
    if(req.clock)check(d.hz>=req.clock,'主频 '+clock(req.clock),typeof d.hz!=='number');
    if(req.ram)check(d.ra>=req.ram,'RAM '+memory(req.ram),typeof d.ra!=='number');
    if(req.flash)check(d.fl>=req.flash,'Flash '+memory(req.flash),typeof d.fl!=='number');
    if(req.pins)check(Number(d.pin)===req.pins,req.pins+' 引脚',d.pin===''||d.pin===undefined);
    if(req.pinsMin)check(Number(d.pin)>=req.pinsMin,req.pinsMin+' 引脚以上',d.pin===''||d.pin===undefined);
    if(req.pinsMax)check(Number(d.pin)<=req.pinsMax,req.pinsMax+' 引脚以内',d.pin===''||d.pin===undefined);
    if(req.fpu)check(d.fpu==='yes','FPU',!d.fpu||d.fpu==='unknown');
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
    if(req.core&&!coreText.includes(String(req.core).toLowerCase()))return false;
    if(req.coreOnly&&!(d.cc===1&&!/[+/,]/.test(coreText)))return false;
    if(req.coreCount&&d.cc!==req.coreCount)return false;
    if(req.micropython&&d.pt!=='micropython_mcu')return false;
    if(req.exact&&!String(d._q||'').includes(req.exact))return false;
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
  function aiApplySoftSignals(d,req,score,reasons){
    const prefs=Array.isArray(req.preferences)?req.preferences:[],softMinimums=req.softMinimums||{},softTargets=req.softTargets||{},softExcluded=Array.isArray(req.softExcludedFeatures)?req.softExcludedFeatures:[],add=(delta,label)=>{score+=delta;if(label)reasons.push(label)};
    if(prefs.includes('lowPower')){if(d.pt==='low_power_mcu')add(12,'低功耗系列');else if(d.pt==='general_purpose_mcu'&&d.hz&&d.hz<=80000000)add(4,'较低运行功耗倾向');else if(d.hz&&d.hz>200000000)add(-6,'频率较高')}
    if(prefs.includes('highPerformance')){if(d.pt==='high_performance_mcu')add(10,'高性能系列');if(d.hz)add(Math.min(8,Math.round(d.hz/100000000)),'主频性能优先')}
    if(prefs.includes('compact')){const pins=Number(d.pin);if(pins&&pins<=32)add(10,'小封装倾向');else if(pins&&pins<=48)add(6,'封装尺寸倾向');else if(pins>100)add(-4,'引脚数偏多')}
    if(prefs.includes('ecosystem')){if((d.boards||[]).length)add(8,'开发板生态');else if((d.parts||[]).length)add(3,'订货信息完整')}
    if(prefs.includes('domestic')&&['HPMicro','Qinheng','GigaDevice','Geehy','MindMotion','Nuvoton','Puya','STC','Allwinner'].includes(d.m))add(10,'国产厂商优先')
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
    let scored=direct.map(d=>{let score=(Number(d.idx)||0)*.38+(Number(d.cov)||0)*.16+((d.parts||[]).length?4:0);const reasons=[];if(req.vendor){if(d.m===req.vendor){score+=18;reasons.push('厂商匹配')}else score-=14}if(req.core){if(String(d.c||d.a||'').toLowerCase().includes(String(req.core).toLowerCase())){score+=15;reasons.push('核心匹配')}else score-=12}if(req.micropython){if(d.pt==='micropython_mcu'){score+=18;reasons.push('MicroPython 生态')}else score-=20}if(req.exact&&String(d._q||'').includes(req.exact)){score+=35;reasons.push('型号命中')}if(req.clock){if(d.hz>=req.clock){score+=12;reasons.push(clock(d.hz)+' 达标')}else if(d.hz)score-=18;else reasons.push('主频未核验')}if(req.ram){if(d.ra>=req.ram){score+=8;reasons.push(memory(d.ra)+' RAM')}else if(d.ra)score-=14;else reasons.push('RAM 未核验')}if(req.flash){if(d.fl>=req.flash){score+=6;reasons.push(memory(d.fl)+' Flash')}else if(d.fl)score-=10;else reasons.push('Flash 未核验')}if(req.fpu){if(d.fpu==='yes'){score+=10;reasons.push('FPU')}else if(d.fpu==='no')score-=18;else reasons.push('FPU 未核验')}if(req.pins){if(Number(d.pin)===req.pins){score+=5;reasons.push(req.pins+' 引脚')}else if(d.pin)score-=4}if(req.pinsMin){if(Number(d.pin)>=req.pinsMin){score+=5;reasons.push(req.pinsMin+' 引脚以上')}else if(d.pin)score-=4}if(req.pinsMax){if(Number(d.pin)<=req.pinsMax){score+=5;reasons.push(req.pinsMax+' 引脚以内')}else if(d.pin)score-=4}Object.entries(req.minimums).forEach(([key,min])=>{const got=aiMetric(d,key);const label=key==='serial'?'串口':key==='usbAny'?'USB':(peripheralByKey.get(key)?.label||key);if(typeof got==='number'&&got>=min){score+=10;reasons.push(label+' '+got+' 路')}else if(typeof got==='number')score-=12;else reasons.push(label+' 未核验')});const soft=aiApplySoftSignals(d,req,score,reasons);score=soft.score;reasons.splice(0,reasons.length,...soft.reasons);if(!reasons.length)reasons.push('选型指数 '+value(d.idx));return {device:d,score:Math.max(0,Math.min(100,Math.round(score))),reasons:reasons.slice(0,5)}}).sort((a,b)=>b.score-a.score||((b.device.idx||0)-(a.device.idx||0))||natural(a.device.n,b.device.n));
    scored=scored.map(item=>{const evaluation=aiEvaluate(item.device,req);const fail=evaluation.failures.map(label=>'不满足 '+label),unknown=evaluation.unknowns.map(label=>'未核验 '+label);const penalty=fail.length*24+unknown.length*14;return {...item,strict:evaluation.strict,violations:fail,unknowns:unknown,score:Math.max(0,Math.min(100,item.score-penalty)),reasons:[...fail,...unknown,...item.reasons].slice(0,5)}}).sort((a,b)=>b.score-a.score||((b.device.idx||0)-(a.device.idx||0))||natural(a.device.n,b.device.n));
    const selected=aiDiverseSelect(scored,req);
    const known=Object.keys(req.minimums).length+Object.keys(req.maximums).length+Object.keys(req.softMinimums||{}).length+Object.keys(req.softTargets||{}).length+Number(Boolean(req.vendor))+Number(Boolean(req.core))+Number(Boolean(req.coreCount))+Number(Boolean(req.clock))+Number(Boolean(req.clockTarget))+Number(Boolean(req.ram))+Number(Boolean(req.ramTarget))+Number(Boolean(req.flash))+Number(Boolean(req.flashTarget))+Number(Boolean(req.pins))+Number(Boolean(req.pinsMin))+Number(Boolean(req.pinsMax))+Number(Boolean(req.fpu))+Number(Boolean(req.micropython))+req.profiles.length+req.preferences.length;const scope=known?'按自然语言理解出的场景、偏好与硬约束排序':'按本地模型与数据完整度排序';const warning=req.warnings.length?' '+req.warnings.join(' '):'';const text=relaxed?'没有找到全部满足且字段已核验的器件，以下仅列出同一核心 / 厂商范围内违约项最少的近似候选；请优先查看“未核验 / 不满足”提示。'+warning:`${scope}，给出 ${selected.length} 款候选。${warning}`;return {req,relaxed,text,results:selected};
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
    if(source.core)parts.push(source.core);
    if(source.coreCount)parts.push(source.coreCount+' 核');
    if(source.coreOnly)parts.push('单核');
    if(source.clock)parts.push('主频 ≥ '+clock(source.clock));
    if(source.ram)parts.push('RAM ≥ '+memory(source.ram));
    if(source.flash)parts.push('Flash ≥ '+memory(source.flash));
    if(source.pins)parts.push(source.pins+' 引脚');
    if(source.pinsMin)parts.push('引脚 ≥ '+source.pinsMin);
    if(source.pinsMax)parts.push('引脚 ≤ '+source.pinsMax);
    if(source.fpu)parts.push('FPU');
    if(source.micropython)parts.push('MicroPython');
    if(source.clockTarget)parts.push('主频约 '+clock(source.clockTarget));
    if(source.ramTarget)parts.push('RAM约 '+memory(source.ramTarget));
    if(source.flashTarget)parts.push('Flash约 '+memory(source.flashTarget));
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
  function restoreAssistant(){return readStoredArray('mcul_assistant_history').filter(message=>message&&typeof message==='object').map(message=>({...message,req:message.req?{...message.req,excludedVendors:message.req.excludedVendors||[],minimums:message.req.minimums||{},maximums:message.req.maximums||{},excludedFeatures:message.req.excludedFeatures||[],softExcludedFeatures:message.req.softExcludedFeatures||[],vagueFeatures:message.req.vagueFeatures||[]}:null,results:(Array.isArray(message.results)?message.results:[]).filter(item=>item&&item.id).map(item=>{const device=byId.get(item.id);return device?{...item,device}:null}).filter(Boolean)}))}
  function renderAssistant(){
    const messages=state.assistantMessages.length?state.assistantMessages:[{role:'assistant',text:'告诉我你的资源约束，我会从当前离线目录中给出可核对的候选。',req:null,results:[]}];
    $('#view').innerHTML=`<div class="assistant-heading page-heading"><div><h1>选型助手 <span class="ai-badge">AI</span></h1><p>本地轻量模型 v${esc(localModel.version)} · 目录约束离线核验</p></div><button id="assistant-reset" class="assistant-reset" title="清空对话">清空</button></div><div class="assistant-shell"><div class="assistant-messages" id="assistant-messages">${messages.map(aiMessageHtml).join('')}</div><div class="assistant-quick"><button data-ai-prompt="需要 120MHz 以上、2 个 UART、CAN、64KB RAM 的 Cortex-M4">Cortex-M4 + CAN</button><button data-ai-prompt="需要 Wi-Fi、蓝牙和 USB，优先低功耗">Wi-Fi + 蓝牙</button><button data-ai-prompt="MicroPython，至少 2 个串口，带摄像头接口">MicroPython + 摄像头</button></div><form class="assistant-composer" id="assistant-form"><textarea id="assistant-input" rows="2" placeholder="例如：需要 120MHz、2 个 UART、CAN、64KB RAM 的 Cortex-M4"></textarea><button class="assistant-send" type="submit">生成候选 <span>↵</span></button></form></div>`;
    const form=$('#assistant-form'),input=$('#assistant-input'),messagesEl=$('#assistant-messages');form.onsubmit=e=>{e.preventDefault();const prompt=input.value.trim();if(!prompt)return;const result=aiRecommend(prompt);state.assistantMessages.push({role:'user',text:prompt},{role:'assistant',text:result.text,req:result.req,results:result.results});saveAssistant();renderAssistant()};input.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')form.requestSubmit()};$('#assistant-reset').onclick=()=>{state.assistantMessages=[];try{localStorage.removeItem('mcul_assistant_history')}catch(_){}renderAssistant()};document.querySelectorAll('[data-ai-prompt]').forEach(button=>button.onclick=()=>{input.value=button.dataset.aiPrompt;input.focus()});document.querySelectorAll('[data-ai-device]').forEach(card=>{card.onclick=()=>openDetail(card.dataset.aiDevice);card.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openDetail(card.dataset.aiDevice)}}});document.querySelectorAll('[data-ai-compare]').forEach(button=>button.onclick=e=>{e.stopPropagation();toggleCompare(button.dataset.aiCompare);button.textContent=state.compare.has(button.dataset.aiCompare)?'已对比':'＋ 对比';button.classList.toggle('selected',state.compare.has(button.dataset.aiCompare))});if(messagesEl)messagesEl.scrollTop=messagesEl.scrollHeight;updateNav();
  }
  function renderCompare(){
    const list=[...state.compare].map(id=>byId.get(id)).filter(Boolean);if(!list.length){$('#view').innerHTML='<div class="page-heading"><h1>参数对比</h1><p>最多同时比较四款器件。</p></div><div class="empty"><strong>尚未加入对比</strong>在器件详情或搜索结果中点击“对比”。</div>';updateNav();return}
    const numeric=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
    const sumKnown=(...values)=>values.some(v=>numeric(v)!==null)?values.reduce((sum,v)=>sum+(numeric(v)??0),0):null;
    const row=(label,display,metric)=>{const scores=metric?list.map(metric).map(numeric):[];const known=scores.filter(v=>v!==null);const best=known.length>=2&&new Set(known).size>1?Math.max(...known):null;return `<tr><td>${label}</td>${list.map((d,i)=>`<td class="${best!==null&&scores[i]===best?'compare-best':''}">${esc(display(d))}</td>`).join('')}</tr>`};
    $('#view').innerHTML=`<div class="page-heading"><h1>参数对比</h1><p>${list.length} / 4 款器件，绿色标出存在差异且数值领先的项目。</p></div><div class="compare-scroll"><table class="compare-table"><thead><tr><th>项目</th>${list.map(d=>`<th>${esc(d.n)}<button class="remove-compare" data-remove="${esc(d.id)}">移出</button></th>`).join('')}</tr></thead><tbody>${row('厂商',d=>d.m)}${row('目录',d=>d.s+' › '+d.l)}${row('核心',d=>d.c||d.a||'—')}${row('FPU',d=>d.fpu==='yes'?'有':d.fpu==='no'?'无':'—',d=>d.fpu==='yes'?1:d.fpu==='no'?0:null)}${row('最高主频',d=>clock(d.hz),d=>d.hz)}${row('Flash',d=>memory(d.fl),d=>d.fl)}${row('RAM',d=>memory(d.ra),d=>d.ra)}${row('TIM',d=>value(d.tim)+(d.tw?' × '+d.tw+' bit':''),d=>d.tim)}${row('ADC 转换器单元',d=>value(d.adcu),d=>d.adcu)}${row('ADC 通道（含内部）',d=>value(d.adch),d=>d.adch)}${row('来源 ADC 原始参数',d=>value(d.adc),d=>d.adc)}${row('GPIO',d=>value(d.gpio),d=>d.gpio)}${row('SERCOM / FLEXCOM',d=>value(d.sercom),d=>d.sercom)}${row('SPI / I²C',d=>value(d.spi)+' / '+value(d.i2c),d=>sumKnown(d.spi,d.i2c))}${row('USART / UART',d=>value(d.usart)+' / '+value(d.uart),d=>sumKnown(d.usart,d.uart))}${row('CAN',d=>value(d.can),d=>d.can)}${row('选型指数',d=>value(d.idx)+' / 100',d=>d.idx)}${row('数据覆盖率',d=>value(d.cov)+'%',d=>d.cov)}${row('完整订货号',d=>(d.parts||[]).length,d=>(d.parts||[]).length)}</tbody></table></div>`;document.querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{state.compare.delete(b.dataset.remove);saveCompare();renderCompare()});updateNav();
    const compareBody=$('.compare-table tbody');
    if(compareBody)compareBody.insertAdjacentHTML('beforeend',row('USB（通用 / Device / Host）',d=>value(d.usb)+' / '+value(d.usbd)+' / '+value(d.usbh),d=>sumKnown(d.usb,d.usbd,d.usbh)));
  }
  function renderData(){
    $('#view').innerHTML=`<div class="page-heading"><h1>数据与版本</h1><p>当前目录快照、可信度规则和厂商覆盖情况。</p></div><div class="info-banner"><b>离线快照 ${esc(catalog.meta.snapshot)}</b><p>全库评分字段平均覆盖率已通过 90% 构建门槛。缺失能力仍显示“—”，不会通过同系列型号自行补齐。</p></div><div class="summary-strip">${summary('平均覆盖率',catalog.meta.averageCoverage+'%')}${summary('覆盖率 ≥ 90%',count(catalog.meta.devicesAt90))}${summary('FPU 已核验',catalog.meta.fpuCoverage+'%')}</div><div class="summary-strip">${summary('系列大类',count(catalog.meta.series))}${summary('产品线',count(catalog.meta.productLines))}${summary('器件变体',count(catalog.meta.devices))}</div><div class="data-panel"><h3>覆盖范围</h3><table class="coverage-table"><thead><tr><th>厂商</th><th>系列</th><th>产品线</th><th>变体</th><th>订货号</th></tr></thead><tbody>${catalog.coverage.map(c=>`<tr><td>${esc(c.m)}</td><td>${count(c.series)}</td><td>${count(c.lines)}</td><td>${count(c.devices)}</td><td>${count(c.parts)}</td></tr>`).join('')}</tbody></table></div><div class="data-panel"><h3>可信度规则</h3><p class="good">● 完整订货号只收录有官方来源的记录，不通过后缀排列组合生成。</p><p>● MCUS 选型指数用于候选排序，不是 CoreMark、DMIPS 或 ULPMark 实测成绩。</p><p>● FPU 的“有 / 无”都必须来自处理器元数据、官方目标能力宏或明确的核心架构事实。</p><p>● 厂商加速器保留原名；没有逐器件完成确认的 Chrom-ART、Neural-ART 等标为待核验候选。</p><p class="caution">● 当前数据是已导入范围，不代表所有厂商完整在售目录已经全部完成。</p></div><div class="data-panel"><h3>应用版本</h3><p>MCUS Android ${esc(catalog.meta.version)} · 现代浅色界面版<br>生成时间：${esc(catalog.meta.generated)}</p></div>`;updateNav();
  }
  function renderAuthor(){
    $('#view').innerHTML=`<div class="page-heading"><h1>关于 MCUS</h1><p>一个面向工程师的离线 MCU 选型目录。</p></div><div class="author-card"><div class="chip-logo">M</div><h2>作者：new.bmp</h2><p>MCUS 汇总厂商 MCU、器件变体、外设资源、核心能力与官方订货号，帮助工程师快速筛选和比较。</p><p>当前版本：${esc(catalog.meta.version)} · 数据器件：${count(catalog.meta.devices)}</p><a class="author-link" href="https://github.com/new-bmp/MCUS">项目主页<br>https://github.com/new-bmp/MCUS ↗</a></div>`;
    updateNav();
  }
  function spec(v,label){return `<div class="spec-cell"><b>${esc(v)}</b><span>${esc(label)}</span></div>`}
  function inventorySection(d){
    const items=d.pi||[];
    const categoryLabels={timing:'定时与控制',analog:'模拟外设',gpio:'GPIO 与中断',connectivity:'通信接口',wireless:'无线连接',memory_bus:'DMA 与外部总线',display_multimedia:'显示与多媒体',security:'安全',accelerator:'计算加速',clock:'时钟',power:'电源与低功耗',system:'系统资源',other:'其他来源特征'};
    const order=['timing','analog','gpio','connectivity','wireless','memory_bus','display_multimedia','security','accelerator','clock','power','system','other'];
    if(!items.length)return `<div class="detail-section"><h2>来源外设清单</h2><div class="feature-panel"><div class="feature-label">当前 CMSIS Pack 没有提供可展开的外设特征；不代表芯片没有外设。</div></div></div>`;
    const grouped=new Map();items.forEach(item=>{const key=item.g||'other';if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(item)});
    return `<div class="detail-section"><h2>来源外设清单 · ${items.length}</h2><div class="inventory-note">这里逐项展示来源明确列出的资源。ADC 以转换器单元和通道为选型参数，不统计 ADC 引脚数量。</div>${order.filter(key=>grouped.has(key)).map(key=>`<div class="inventory-group"><h3>${categoryLabels[key]||key}</h3><div class="inventory-list">${grouped.get(key).map(item=>`<div class="inventory-item"><b>${esc(item.n)}</b><span>${esc(item.d)}</span></div>`).join('')}</div></div>`).join('')}</div>`;
  }
  function quoteEndpoint(){
    const configured=String(window.MCUS_QUOTE_API||'').trim();
    if(configured)return configured;
    if(location.protocol==='https:'||location.protocol==='http:')return new URL('/api/quotes',location.href).href;
    return '';
  }
  function safeHttpUrl(value){try{const url=new URL(value);return url.protocol==='https:'||url.protocol==='http:'?url.href:''}catch(_){return ''}}
  function quoteMessage(code,fallback){
    const messages={not_configured:'淘宝询价服务尚未配置，请先在 Cloudflare Worker 设置淘宝开放平台参数。',invalid_part:'该订货号不适合直接询价。',taobao_api_error:'淘宝接口暂时不可用，请稍后重试。',no_strict_matches:'没有找到标题精确包含该订货号的芯片商品。',network_error:'无法连接询价服务，请检查网络后重试。'};
    return messages[code]||fallback||'询价失败，请稍后重试。';
  }
  function quotePanelHtml(part,data){
    const quotes=Array.isArray(data.quotes)?data.quotes:[];
    const updated=data.updatedAt?new Date(data.updatedAt).toLocaleString('zh-CN',{hour12:false}):'刚刚';
    return `<div class="quote-head"><div><b>淘宝询价 · ${esc(part)}</b><br><span>${quotes.length} 家严格匹配店铺 · ${esc(updated)}</span></div></div>${quotes.length?`<div class="quote-list">${quotes.map(item=>{const link=safeHttpUrl(item.url);const price=Number(item.price);return `<div class="quote-row"><div><span class="quote-shop">${esc(item.shop||'淘宝商家')}</span><p class="quote-title">${esc(item.title||part)}</p></div><div class="quote-price"><b>¥${Number.isFinite(price)?price.toFixed(2):'—'}</b>${link?`<a href="${esc(link)}">查看商品 ↗</a>`:''}</div></div>`}).join('')}</div>`:`<div class="quote-status">${esc(quoteMessage('no_strict_matches'))}</div>`}<p class="quote-note">仅展示标题精确包含完整订货号、价格有效且来自不同店铺的芯片商品；已排除开发板、核心板、模块、套件、烧录器和二手拆机商品。严格结果不足 3 家时不会用模糊型号补足，实际成交价以淘宝结算页为准。</p>`;
  }
  async function requestQuotes(part){
    const panel=$('#quote-panel');if(!panel)return;
    part=String(part||'').trim().toUpperCase();
    if(!/^[A-Z0-9][A-Z0-9+._\/-]{3,63}$/.test(part)){panel.hidden=false;panel.innerHTML=`<div class="quote-head"><b>淘宝询价</b></div><div class="quote-status">${esc(quoteMessage('invalid_part'))}</div>`;return}
    document.querySelectorAll('[data-quote-part]').forEach(button=>button.classList.toggle('active',button.dataset.quotePart===part));
    panel.hidden=false;panel.dataset.part=part;panel.innerHTML=`<div class="quote-head"><b>淘宝询价 · ${esc(part)}</b></div><div class="quote-status">正在查找精确型号的芯片商品…</div>`;
    panel.scrollIntoView({behavior:'smooth',block:'nearest'});
    const endpoint=quoteEndpoint();
    if(!endpoint){panel.innerHTML=`<div class="quote-head"><b>淘宝询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage('not_configured'))}</div>`;return}
    if(quoteAbort)quoteAbort.abort();quoteAbort=new AbortController();
    try{
      const url=new URL(endpoint,location.href);url.searchParams.set('part',part);
      const response=await fetch(url.href,{headers:{accept:'application/json'},signal:quoteAbort.signal});
      const data=await response.json().catch(()=>({}));
      if(panel.dataset.part!==part)return;
      if(!response.ok)throw Object.assign(new Error(data.message||''),{code:data.code||'taobao_api_error'});
      panel.innerHTML=quotePanelHtml(part,data);
    }catch(error){
      if(error&&error.name==='AbortError')return;
      if(panel.dataset.part!==part)return;
      const code=error&&error.code?error.code:'network_error';
      panel.innerHTML=`<div class="quote-head"><b>淘宝询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage(code,error&&error.message))}<br><button class="quote-retry" data-quote-retry="${esc(part)}">重新询价</button></div>`;
      const retry=panel.querySelector('[data-quote-retry]');if(retry)retry.onclick=()=>requestQuotes(retry.dataset.quoteRetry);
    }
  }
  function openDetail(id){
    state.detail=id;const d=byId.get(id);if(!d)return;const features=[...(d.acc||[]),...(d.feat||[])];const parts=d.parts||[];const selected=state.compare.has(d.id);const layer=$('#detail-layer');
    layer.innerHTML=`<div class="detail-backdrop"></div><section class="detail-page"><div class="detail-header"><button class="back-btn">‹</button><div class="detail-title"><h1>${esc(d.n)}</h1><p>${esc(d.m)} · ${esc(d.s)} · ${esc(productType(d.pt))}</p></div><div class="detail-score"><b>${value(d.idx)}</b><span>选型指数 / 100</span></div></div><div class="detail-hero"><div class="detail-path">${esc(d.m)} › ${esc(d.s)} › ${esc(d.l)} › ${esc(productType(d.pt))}</div><div class="detail-model"><b>${esc(d.n)}</b><span>变体码 ${esc(d.v||'—')}</span></div></div><div class="detail-section"><h2>核心与存储</h2><div class="spec-grid">${spec(d.c||d.a||'—','处理器核心')}${spec((d.cc||1)+' core','核心数')}${spec(clock(d.hz),'最高核心频率')}${spec(memory(d.fl),'Flash')}${spec(memory(d.ra),'RAM')}${spec(d.fpu==='yes'?'有':d.fpu==='no'?'无':'—','FPU')}</div></div><div class="detail-section"><h2>外设资源</h2><div class="spec-grid">${spec(value(d.tim)+(d.tw?' × '+d.tw+'b':''),'TIM 数量 / 位宽')}${spec(value(d.adcu),'ADC 转换器单元')}${spec(value(d.adch),'ADC 通道（含内部）')}${spec(value(d.adc)+(d.adr?' · '+d.adr+' bit':''),'来源 ADC 原始参数')}${spec(value(d.gpio),'GPIO')}${spec(value(d.sercom),'SERCOM / FLEXCOM')}${spec(value(d.spi),'SPI')}${spec(value(d.i2c),'I²C')}${spec(value(d.usart),'USART')}${spec(value(d.uart),'UART')}${spec(value(d.can),'CAN')}${spec(value(d.usbd),'USB Device')}${spec(value(d.usbh),'USB Host')}${spec(value(d.eth),'Ethernet')}${spec(d.pin||'—','引脚数')}</div></div>${inventorySection(d)}<div class="detail-section"><h2>厂商加速器与特性</h2><div class="feature-panel">${features.length?`<div class="feature-label">来源已经确认的能力</div><div class="feature-list">${features.map(x=>`<span class="feature-chip confirmed">${esc(x)}</span>`).join('')}</div>`:'<div class="feature-label">当前来源没有已确认的专用加速器记录</div>'}${(d.pending||[]).length?`<div class="feature-label" style="margin-top:10px">待逐器件官方文档核验</div><div class="feature-list">${d.pending.map(x=>`<span class="feature-chip pending">${esc(x)}</span>`).join('')}</div><p class="feature-note">候选项不代表该具体后缀型号已经确认支持。</p>`:''}</div></div><div class="detail-section"><h2>评分拆解</h2><div class="score-panel">${[['计算',d.cs],['存储',d.ms],['外设',d.ps],['加速器',d.acs]].map(x=>`<div class="score-row"><label>${x[0]}</label><div class="score-bar"><i style="width:${Math.max(0,Math.min(100,x[1]||0))}%"></i></div><b>${value(x[1])}</b></div>`).join('')}<p class="score-note">数据覆盖率 ${value(d.cov)}%。这是选型排序指标，不是实测性能。CoreMark：${value(d.cm)} · DMIPS：${value(d.dm)}。</p></div></div><div class="detail-section"><h2>官方完整订货号 · ${parts.length}</h2>${parts.length?`<div class="parts">${parts.map(p=>`<div class="part-row"><div><b>${esc(p.n)}</b><p>后缀 ${esc(p.s||'—')} · 封装码 ${esc(p.p||'—')} · 温度码 ${esc(p.t||'—')} · 包装 ${esc(p.k||'—')}</p></div><div class="part-actions"><span class="verified">✓ 已核验</span><button class="quote-trigger" data-quote-part="${esc(p.n)}">询价</button></div></div>`).join('')}</div>`:'<div class="feature-panel"><div class="feature-label">完整订货号尚未导入；不会根据后缀组合自动生成。请明确输入厂商完整订货号后询价。</div><form class="quote-manual" id="quote-manual"><input id="quote-manual-part" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="输入完整订货号，如 STM32F429ZIT6"><button type="submit">询价</button></form></div>'}<div class="quote-panel" id="quote-panel" hidden aria-live="polite"></div></div><button class="detail-action primary" id="detail-compare">${selected?'✓ 已加入对比':'＋ 加入参数对比'}</button>${d.src?`<a class="detail-action" href="${esc(d.src)}">打开来源页面 ↗</a>`:''}</section>`;
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
    requestAnimationFrame(()=>layer.classList.add('open'));layer.querySelector('.back-btn').onclick=closeDetail;layer.querySelector('.detail-backdrop').onclick=closeDetail;$('#detail-compare').onclick=()=>{toggleCompare(d.id);openDetail(d.id)};layer.querySelectorAll('[data-quote-part]').forEach(button=>button.onclick=()=>requestQuotes(button.dataset.quotePart));const manualForm=layer.querySelector('#quote-manual');if(manualForm)manualForm.onsubmit=event=>{event.preventDefault();requestQuotes(layer.querySelector('#quote-manual-part').value)};
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
