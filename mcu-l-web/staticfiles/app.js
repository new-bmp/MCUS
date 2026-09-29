(function(){
  'use strict';
  const APP_VERSION='1.5';
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
    if(Number(v)===0)return '0 B';
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
  const quoteProvider=String(window.MCUS_QUOTE_PROVIDER||'lcsc').toLowerCase()==='ickey'?'ickey':'lcsc';
  const quotesEnabled=window.MCUS_QUOTES_ENABLED===true||window.MCUS_LCSC_QUOTES_ENABLED===true;
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
  devices.forEach(d=>{const peripheralText=(d.pi||[]).flatMap(item=>[item.n,item.t,item.d]).join(' ');const aliases=peripheralFilters.filter(item=>peripheralCount(d,item.key)).map(item=>item.aliases).join(' ');const vendorAliases=d.m==='Nationz'?'国民技术 国民 nationz nations nsing n32':d.m==='MindMotion'?'灵动 灵动微电子 灵动微 mindmotion mm32':d.m==='Qinheng'?'沁恒 wch qinheng nanjing qinheng microelectronics qingke 青稞':d.m==='STC'?'stc 宏晶 hongjing stc microelectronics 8051':d.m==='HPMicro'?'先楫 hpm hpmicro risc-v 上海先楫':d.m==='Renesas'?'瑞萨 renesas ra rx rl78 rh850 synergy risc-v 瑞萨电子':d.m==='Artery'?'雅特力 artery arterytek at32 at32f at32a at32l at32m at32wb':d.m==='Allwinner'?'全志 allwinner xradio 芯之联 wireless mcu 实时 异构 soc 数传':d.m==='MicroPy MCU'?'micropy micropython mpy raspberry pi rp2040 rp2350 kendryte k210 micropython mcu':d.m==='Texas Instruments'?'德州仪器 texas instruments ti c2000 tms320 c28x dsp 实时控制器':' ';d._q=[d.n,d.l,d.s,d.f,d.m,d.pt,vendorAliases,d.v,d.c,d.a,peripheralText,aliases,...(d.boards||[]),...(d.parts||[]).map(p=>p.n)].join(' ').toLowerCase()});
  function readStoredArray(key){
    try{const parsed=JSON.parse(localStorage.getItem(key)||'[]');return Array.isArray(parsed)?parsed:[]}
    catch(_){try{localStorage.removeItem(key)}catch(__){}return []}
  }
  function readStoredString(key){try{return String(localStorage.getItem(key)||'').trim()}catch(_){return ''}}
  function writeStoredString(key,value){try{if(value)localStorage.setItem(key,value);else localStorage.removeItem(key);return true}catch(_){return false}}
  function writeStored(key,value){try{localStorage.setItem(key,JSON.stringify(value));return true}catch(_){return false}}
  const state={tab:'catalog',query:'',vendorFilter:'',coreFilter:'',peripheralFilter:'',peripheralMin:1,sort:'score',limit:120,detail:null,browse:{vendor:null,series:null,line:null},compare:new Set(readStoredArray('mcul_compare').filter(id=>byId.has(id)))};
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
  const vendorLogoFiles={Allwinner:'allwinner.png',Artery:'artery.svg',Espressif:'espressif.svg',Geehy:'geehy.ico',GigaDevice:'gigadevice.svg',HPMicro:'hpmicro.png',Infineon:'infineon.svg',Microchip:'microchip.ico',MindMotion:'mindmotion.png',Nationz:'nationz.png',Nuvoton:'nuvoton.jpg',Puya:'puya.ico',Qinheng:'qinheng.svg',Renesas:'renesas.svg',STC:'stc.svg',STMicroelectronics:'stmicroelectronics.svg','Texas Instruments':'texas-instruments.ico'};
  function vendorLogo(name){const file=vendorLogoFiles[name];const cssName=name.toLowerCase().replace(/[^a-z0-9]+/g,'-');return `<span class="folder-icon vendor logo-${cssName}"><span class="vendor-fallback">${esc(vendorGlyph(name))}</span>${file?`<img src="vendor-${esc(file)}" alt="${esc(name)} Logo" onerror="this.remove()">`:''}</span>`}
  function vendorName(name){if(name==='Allwinner')return '全志（Allwinner / XRadio）';if(name==='Artery')return '雅特力（Artery / ArteryTek）';if(name==='Microchip')return 'Microchip（原 Atmel）';if(name==='Qinheng')return '沁恒（WCH）';if(name==='Renesas')return '瑞萨电子（Renesas）';if(name==='STC')return 'STC（宏晶）';if(name==='HPMicro')return '先楫半导体（HPMicro）';if(name==='Nationz')return '国民技术（Nationz）';if(name==='MindMotion')return '灵动微电子（MindMotion / MM32）';return name}
  function productType(type){return ({wireless_mcu:'无线 MCU',wireless_audio_mcu_soc:'无线音频 MCU SoC',wireless_connectivity_chip:'无线连接芯片',heterogeneous_realtime_soc:'带实时 MCU 核的 SoC',micropython_mcu:'MicroPython MCU',dsp_mcu:'DSP 实时控制器'})[type]||'MCU'}
  function categoryTitle(list){if(list[0]?.m==='Allwinner'||list[0]?.m==='MicroPy MCU')return [...new Set(list.map(d=>productType(d.pt)))].join(' / ');return [...new Set(list.map(d=>d.c||d.a).filter(Boolean))].sort(natural).join(' / ')||'MCU 系列'}
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
    if(quotesEnabled){
      const quoteRows=list.map((d,index)=>{const parts=d.parts||[];const options=parts.map((part,partIndex)=>`<option value="${esc(part.n)}"${partIndex===0?' selected':''}>${esc(part.n)}</option>`).join('');return `<div class="compare-quote-item"><div><b>${esc(d.n)}</b>${parts.length?`<label class="compare-part-label">完整订货号<select data-compare-part-select="compare-quote-${index}">${options}</select></label>`:'<small>没有已核验的完整订货号</small>'}</div>${parts.length?`<button class="quote-trigger" data-compare-quote="compare-quote-${index}" data-compare-panel="compare-quote-panel-${index}">立创查询</button>`:''}<div class="quote-panel compare-quote-panel" id="compare-quote-panel-${index}" hidden aria-live="polite"></div></div>`}).join('');
      $('#view').insertAdjacentHTML('beforeend',`<section class="compare-quotes"><div class="section-heading"><h2>立创商城实时价格</h2><span>精确订货号</span></div><p class="compare-quote-note">仅按每款器件的首个已核验完整订货号查询，不根据型号后缀猜测；价格、库存和阶梯价以立创商城结算页为准。</p>${quoteRows}</section>`);
      document.querySelectorAll('[data-compare-quote]').forEach(button=>button.onclick=()=>{const select=$(`[data-compare-part-select="${button.dataset.compareQuote}"]`);requestQuotes(select?.value||'',1,$(`#${button.dataset.comparePanel}`))});
    }
    const compareBody=$('.compare-table tbody');
    if(compareBody)compareBody.insertAdjacentHTML('beforeend',row('USB（通用 / Device / Host）',d=>engineeringValue(d.usb)+' / '+engineeringValue(d.usbd)+' / '+engineeringValue(d.usbh),d=>sumKnown(d.usb,d.usbd,d.usbh)));
    if(compareBody)compareBody.insertAdjacentHTML('beforeend',row('外部存储总线',d=>Array.isArray(d.eb)?((d.eb||[]).join(' / ')||'无'):'未核验'));
  }
  function lcscSettingsHtml(){
    const endpoint=readStoredString('mcul_lcsc_quote_api')||String(window.MCUS_LCSC_QUOTE_API||'').trim();
    return `<div class="data-panel quote-settings"><h3>立创商城接口地址</h3><p>填写你自己的 HTTPS Worker 或授权 API 网关地址。填写域名会自动使用 /api/quotes 路径。不要填写 API Key，密钥应保存在服务端。</p><form id="lcsc-api-form"><input id="lcsc-api-endpoint" type="url" inputmode="url" autocomplete="url" spellcheck="false" placeholder="https://example.workers.dev" value="${esc(endpoint)}"><button type="submit">保存</button><button type="button" id="lcsc-api-clear">清除</button></form><small id="lcsc-api-status">${endpoint?'已保存自定义地址':'未填写，将使用当前站点的 /api/quotes'}</small></div>`;
  }
  function setupLcscSettings(){
    if(!document.querySelector('.quote-settings'))$('#view').insertAdjacentHTML('beforeend',lcscSettingsHtml());
    const form=$('#lcsc-api-form'),input=$('#lcsc-api-endpoint'),status=$('#lcsc-api-status');if(!form||!input||!status)return;
    const setStatus=(message,kind='')=>{status.textContent=message;status.className=kind};
    form.onsubmit=event=>{event.preventDefault();const raw=input.value.trim();if(!raw){writeStoredString('mcul_lcsc_quote_api','');setStatus('已清除，将使用当前站点的 /api/quotes','ok');return}try{const url=new URL(raw);const local=url.protocol==='http:'&&['localhost','127.0.0.1'].includes(url.hostname);if(url.protocol!=='https:'&&!local)throw new Error('protocol');url.hash='';const normalized=url.href.replace(/\/$/,'');if(!writeStoredString('mcul_lcsc_quote_api',normalized)){setStatus('无法写入本机设置','error');return}input.value=normalized;setStatus('已保存，详情页和对比页将使用该地址','ok')}catch(_){setStatus('地址无效：请输入 HTTPS 地址（本机调试可用 localhost）','error')}};
    $('#lcsc-api-clear').onclick=()=>{input.value='';writeStoredString('mcul_lcsc_quote_api','');setStatus('已清除，将使用当前站点的 /api/quotes','ok')};
  }
  function renderData(){
    $('#view').innerHTML=`<div class="page-heading"><h1>数据与版本</h1><p>当前目录快照、可信度规则和厂商覆盖情况。</p></div><div class="info-banner"><b>离线快照 ${esc(catalog.meta.snapshot)}</b><p>全库评分字段平均覆盖率已通过 90% 构建门槛。缺失能力仍显示“—”，不会通过同系列型号自行补齐。</p></div><div class="summary-strip">${summary('平均覆盖率',catalog.meta.averageCoverage+'%')}${summary('覆盖率 ≥ 90%',count(catalog.meta.devicesAt90))}${summary('FPU 已核验',catalog.meta.fpuCoverage+'%')}</div><div class="summary-strip">${summary('系列大类',count(catalog.meta.series))}${summary('产品线',count(catalog.meta.productLines))}${summary('器件变体',count(catalog.meta.devices))}</div><div class="data-panel"><h3>覆盖范围</h3><table class="coverage-table"><thead><tr><th>厂商</th><th>系列</th><th>产品线</th><th>变体</th><th>订货号</th></tr></thead><tbody>${catalog.coverage.map(c=>`<tr><td>${esc(c.m)}</td><td>${count(c.series)}</td><td>${count(c.lines)}</td><td>${count(c.devices)}</td><td>${count(c.parts)}</td></tr>`).join('')}</tbody></table></div><div class="data-panel"><h3>可信度规则</h3><p class="good">● 完整订货号只收录有官方来源的记录，不通过后缀排列组合生成。</p><p>● MCUS 选型指数用于候选排序，不是 CoreMark、DMIPS 或 ULPMark 实测成绩。</p><p>● FPU 的“有 / 无”都必须来自处理器元数据、官方目标能力宏或明确的核心架构事实。</p><p>● 厂商加速器保留原名；没有逐器件完成确认的 Chrom-ART、Neural-ART 等标为待核验候选。</p><p class="caution">● 当前数据是已导入范围，不代表所有厂商完整在售目录已经全部完成。</p></div><div class="data-panel"><h3>应用版本</h3><p>MCUS Android ${esc(APP_VERSION)} · 现代浅色界面版<br>生成时间：${esc(catalog.meta.generated)}</p></div>`;updateNav();
    setupLcscSettings();
  }
  function renderAuthor(){
    $('#view').innerHTML=`<div class="page-heading"><h1>关于 MCUS</h1><p>一个面向工程师的离线 MCU 选型目录。</p></div><div class="author-card"><div class="chip-logo">M</div><h2>作者：new.bmp</h2><p>MCUS 汇总厂商 MCU、器件变体、外设资源、核心能力与官方订货号，帮助工程师快速筛选和比较。</p><p>当前版本：${esc(APP_VERSION)} · 数据器件：${count(catalog.meta.devices)}</p><a class="author-link" href="https://github.com/new-bmp/MCUS">项目主页<br>https://github.com/new-bmp/MCUS ↗</a></div>`;
    updateNav();
  }
  function spec(v,label){const text=String(v??'—');return `<div class="spec-cell"><b class="spec-value" title="${esc(text)}">${esc(text)}</b><button class="spec-expand" type="button" data-spec-expand aria-expanded="false" hidden>展开</button><span>${esc(label)}</span></div>`}
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
    // Conditions, bank modes and cache-hit claims are shown with the official
    // paragraph in the memory section, never reduced to a guessed scalar.
    const ramRegions=(d?.mem||[]).filter(item=>/(?:ram|sram|tcm|ccm|psram|memory)/i.test(String(item?.n||''))&&!/(?:flash|rom|factory|nonmain)/i.test(String(item?.n||''))).map(item=>item.s?`${item.n} ${memory(item.s)}`:item.n).filter(Boolean);
    if(d?.ramarch)String(d.ramarch).split(';').filter(Boolean).forEach(item=>{if(!ramRegions.some(region=>String(region).toLowerCase().startsWith(item.toLowerCase())))ramRegions.push(item)});
    const ramTypes=[...new Set(ramRegions.map(item=>String(item).split(/\s+/)[0]).filter(Boolean))];
    if(/itcm/i.test(text)&&!ramTypes.some(item=>/itcm/i.test(item)))ramTypes.push('ITCM');
    if(/dtcm/i.test(text)&&!ramTypes.some(item=>/dtcm/i.test(item)))ramTypes.push('DTCM');
    if(/ccm/i.test(text)&&!ramTypes.some(item=>/ccm/i.test(item)))ramTypes.push('CCM');
    if(/axi\s*sram/i.test(text)&&!ramTypes.some(item=>/axi/i.test(item)))ramTypes.push('AXI SRAM');
    return {timerWidths:[...widths].filter(Number.isFinite).sort((a,b)=>a-b),adcResolution:Number(adcResolution)||null,dacResolution:Number(dacResolution)||null,adcRate:directAdcRate||rate('(?:adc|模数转换|采样)'),dacRate:directDacRate||rate('(?:dac|数模转换)'),ioRate:directIoRate||rate('(?:gpio|i/o|io|引脚翻转)'),flashFacts:[...new Set(flashFacts)],ramRegions,ramTypes,cache:String(d?.cache||'').toLowerCase()==='yes'||/i-cache|d-cache|icache|dcache|cache/i.test(text),ramEcc:String(d?.recc||'').toLowerCase()==='yes',dma:d?.dma,externalBus:Array.isArray(d?.eb)?d.eb:[],multiCore:Number(d?.cc)>1,exclusiveRam:String(d?.ramex||'').toLowerCase()==='yes'};
  }
  const memoryHelp={
    flash_banks:'核对具体容量和 Bank 模式。双 Bank、跨 Bank 读写和 Bank 交换是不同能力。',
    flash_latency:'等待周期取决于主频、电压及配置。缓存命中或 ART 加速下的零等待不代表所有 Flash 访问均为零等待。',
    flash_ecc:'核对 ECC 的覆盖区域、纠错位数和编程对齐要求；不要把 Flash ECC 当作 RAM ECC。',
    flash_erase:'用于规划参数存储和固件升级。编程单位、页 / 扇区擦除大小及整片擦除不同。',
    flash_endurance:'擦写次数和保持时间受温度及工作条件影响，选型时请同时核对手册电气特性表。',
    flash_protection:'读保护、写保护和安全区域保护用途不同；部分保护操作不可逆。',
    flash_layout:'代码区、数据 Flash、启动区与 EEPROM 可能是独立区域，也可能共用物理阵列。',
    flash_xip:'区分片内 Flash、封装内 Flash 和外接 Flash。支持 XIP 表示可以从映射的外部存储执行代码。',
    ram_tcm:'TCM / CCM 连接路径与普通系统 SRAM 不同，DMA 或其他核心不一定可访问。',
    ram_ecc:'ECC 与奇偶校验不同；核对覆盖的 RAM 分区，以及上电初始化要求。',
    ram_retention:'低功耗保持受电源域、模式和配置影响，不能由总 RAM 容量推断保持容量。',
    ram_access:'检查核心、DMA、外设主设备的访问权限；专用用途的 RAM 不一定是核心独占 RAM。',
    ram_banks:'RAM 分区及总线连接影响并行访问和 DMA 性能；地址别名不能作为额外容量累加。',
    ram_external:'核对 PSRAM / SDRAM 是否实际集成、封装内可选或需要外接，接口支持不代表容量已安装。',
    memory_cache:'核对 I-Cache / D-Cache 结构和维护要求。DMA 与 CPU 共用数据时要检查一致性。'
  };
  function memoryDocuments(d){return (d?.me||[]).map(key=>catalog.memoryDocuments?.[key]).filter(doc=>doc&&safeHttpUrls(doc.url).length)}
  function memorySourceUrl(doc,page){return doc.url.split('#')[0]+(Number.isInteger(page)&&page>0?'#page='+page:'')}
  function memoryPageLabel(page){return Number.isInteger(page)&&page>0?'PDF 第 '+page+' 页':'官方在线资料'}
  function memoryFactApplies(d,fact){
    const quote=String(fact?.quote||''),line=String(d?.l||'').toLowerCase();
    if(!quote||!line)return true;
    // Family handbooks may contain adjacent tables for several product lines.
    // A fact naming another line must not be shown as if it described the
    // selected exact device; unqualified prose remains usable evidence.
    const names=[...quote.matchAll(/\b(?:stm|mm32|n32|gd32|at32|hpm\d*|ra\d+|rx\d+)[a-z0-9_-]{2,}\b/gi)].map(match=>match[0].toLowerCase());
    if(!names.length)return true;
    const normalized=line.replace(/[^a-z0-9]/g,'');
    return names.some(name=>normalized.includes(name.replace(/[^a-z0-9]/g,'')));
  }
  function memoryEvidenceSection(d){
    const docs=memoryDocuments(d),regions=d?.mem||[],groups=new Map();
    docs.forEach(doc=>(doc.facts||[]).filter(fact=>memoryFactApplies(d,fact)).forEach(fact=>{if(!groups.has(fact.key))groups.set(fact.key,[]);const items=groups.get(fact.key);if(!items.some(x=>x.fact.quote===fact.quote))items.push({doc,fact})}));
    const groupHtml=group=>[...groups.values()].filter(items=>items[0].fact.group===group).map(items=>{
      const {fact}=items[0];
      return `<details class="memory-fact"><summary><b>${esc(fact.title)}</b><span>${items.length} 条原文与条件</span></summary><p class="memory-help">${esc(memoryHelp[fact.key]||'核对手册中的适用型号和条件。')}</p>${items.map(({doc,fact})=>`<div class="memory-evidence"><p>${esc(fact.quote)}</p><small>适用资料：${esc(doc.scope.join(' / '))}</small><a href="${esc(memorySourceUrl(doc,fact.page))}">${esc(doc.title)} · ${memoryPageLabel(fact.page)} ↗</a></div>`).join('')}</details>`;
    }).join('');
    const figures=docs.flatMap(doc=>(doc.figures||[]).filter(figure=>!figure.lines||figure.lines.includes(d.l)).map(figure=>({doc,figure})));
    const diagrams=figures.map(({doc,figure})=>{
      const image=/^memory-arch-[a-f0-9]+-p\d+\.png$/.test(figure.image)?figure.image:'';
      return `<details class="memory-diagram"><summary><b>官方架构图 · ${esc(doc.scope.join(' / '))}</b><span>${memoryPageLabel(figure.page)}</span></summary>${image?`<img src="${esc(image)}" alt="${esc(figure.title)}" loading="lazy" decoding="async">`:''}<p>${esc(figure.title)}</p><a href="${esc(memorySourceUrl(doc,figure.page))}">打开官方原图与说明 ↗</a></details>`;
    }).join('');
    const address=regions.length?`<details class="memory-address"><summary><b>存储地址与区域</b><span>${regions.length} 条来源记录</span></summary><p class="memory-help">名称和地址保留厂商来源。区域可能重叠或互为别名，不重复计入总容量。读写执行权限不表示 DMA 或多核共享权限。</p><div class="memory-table-wrap"><table><thead><tr><th>区域 / 类型</th><th>容量</th><th>起始地址</th><th>访问 / 核心</th></tr></thead><tbody>${regions.map(r=>`<tr><td>${esc(r.n)}${r.type?`<small>${esc(r.type)}</small>`:''}${r.alias?`<small>别名 → ${esc(r.alias)}</small>`:''}</td><td>${r.s?esc(memory(r.s)):'未披露'}</td><td>${esc(r.start||'未披露')}</td><td>${esc([r.a,r.core].filter(Boolean).join(' / ')||'未披露')}</td></tr>`).join('')}</tbody></table></div></details>`:'';
    if(!docs.length&&!regions.length)return detailAccordion('Flash / RAM 工程属性','<p class="memory-help">尚无可核验的存储章节或地址记录。</p>',false,'未收录');
    const body=`<p class="memory-help">按该型号关联的官方资料整理，展开查看原文、适用型号和页码。系列手册中的容量、配置或核心差异请以条目条件为准。</p>${address}${['flash','ram','system'].map((g,i)=>{const content=groupHtml(g);return content?`<h3 class="detail-subheading">${['Flash 属性','RAM 属性','Cache 与存储系统'][i]}</h3>${content}`:''}).join('')}${diagrams?`<h3 class="detail-subheading">官方架构图</h3>${diagrams}`:''}`;
    return detailAccordion('Flash / RAM 工程属性',body,false,`${groups.size} 类属性 · ${figures.length} 张图`);
  }
  function compactFlashQuote(value,max=96){
    const text=String(value||'').replace(/\s+/g,' ').trim();
    return text.length>max?text.slice(0,max-1)+'…':text;
  }
  function flashEvidence(d){
    return memoryDocuments(d).flatMap(doc=>(doc.facts||[]).filter(fact=>fact?.group==='flash'&&memoryFactApplies(d,fact)).map(fact=>({doc,fact})));
  }
  function flashUpgradeAssessment(d){
    const items=flashEvidence(d),quotes=items.map(({fact})=>String(fact.quote||''));
    const byKey=key=>items.filter(({fact})=>fact.key===key);
    const first=(key,pattern)=>byKey(key).find(({fact})=>!pattern||pattern.test(String(fact.quote||'')));
    const bankItems=byKey('flash_banks'),bankTextAll=bankItems.map(({fact})=>String(fact.quote||'')).join(' ');
    const conditionalBank=/for dual-bank devices|on dual-bank products|if.{0,24}dual.?bank|depending on|selectable|configurable|可配置|根据型号/i;
    const explicitDual=Number(d?.fb)>=2||bankItems.some(({fact})=>{const q=String(fact.quote||'');return /dual.?bank flash|flash.{0,35}dual.?bank|two flash banks|双\s*bank|双区.{0,12}(?:flash|闪存)/i.test(q)&&!conditionalBank.test(q)});
    const rww=/read.?while.?write|read.?during.?write|simultaneous.{0,18}read.{0,18}write|同时读写/i.test(bankTextAll);
    const swap=/bank swap|boot swap|address swap|\bBFB2\b|bank 交换|区交换/i.test(bankTextAll+' '+quotes.join(' '));
    const erase=first('flash_erase',/\d+\s*(?:bytes?|kbytes?|kb|words?|bits?)|page|sector|block|页|扇区|块/i);
    const endurance=first('flash_endurance',/\b\d[\d,.]*\s*(?:k|m)?\s*(?:cycles?|次)|\b\d+\s*(?:years?|年)|data retention[^.]{0,100}\d/i);
    const protection=byKey('flash_protection').length>0;
    const layout=byKey('flash_layout');
    const boot=layout.some(({fact})=>/boot.?loader|secure boot|boot swap|\bISP\b|\bIAP\b|self.program/i.test(String(fact.quote||'')));
    const ecc=String(d?.fecc||'').toLowerCase()==='yes'||byKey('flash_ecc').length>0;
    const xip=byKey('flash_xip').some(({fact})=>/\bXIP\b|execute.?in.?place|external.{0,30}flash|QSPI|OSPI|OctoSPI|FlexSPI|外部.{0,16}(?:flash|闪存)/i.test(String(fact.quote||'')));
    const capacity=Number(d?.fl)||0;
    let score=0;
    if(capacity>0)score+=10;if(capacity>=262144)score+=10;if(capacity>=524288)score+=5;
    if(explicitDual)score+=20;if(rww||swap)score+=15;if(erase)score+=10;if(protection)score+=10;if(ecc)score+=10;if(boot)score+=5;if(endurance)score+=5;if(xip)score+=5;
    score=Math.min(100,score);
    let label='关键信息不足',level='low',note='先确认擦除粒度、Bootloader 占用、断电恢复和回滚方案。';
    if(explicitDual&&(rww||swap)){label='A/B 升级条件较好';level='high';note='已看到双 Bank 与读写并行或 Bank Swap 证据，仍需核对具体容量配置、启动映射和失败回滚。'}
    else if(explicitDual){label='双 Bank，切换机制待核验';level='medium';note='可规划双镜像，但不能仅凭双 Bank 推断支持无停机写入或自动回滚。'}
    else if(xip&&boot){label='可考虑外部 Flash 暂存';level='medium';note='存在外部 Flash / XIP 与启动相关证据，可评估外部暂存升级包或二级 Bootloader。'}
    else if(capacity>=262144&&boot){label='可评估单区 Bootloader';level='medium';note='容量和启动路径具备基础条件，必须设计断电保护、校验与恢复分区。'}
    const tags=[];if(rww)tags.push('RWW');if(swap)tags.push('Swap');
    const bankText=Number(d?.fb)>0?String(Number(d.fb))+' Bank'+(tags.length?' · '+tags.join(' · '):''):explicitDual?'双 Bank'+(tags.length?' · '+tags.join(' · '):''):tags.length?tags.join(' · '):'未核验';
    return {items,score,label,level,note,dual:explicitDual,rww,swap,ecc,protection,xip,boot,bankText,
      eraseText:erase?compactFlashQuote(erase.fact.quote):'未核验',enduranceText:endurance?compactFlashQuote(endurance.fact.quote):'未核验',
      sourceCount:new Set(items.map(({doc})=>doc.url)).size};
  }
  function flashUpgradeSection(d){
    const a=flashUpgradeAssessment(d);
    const check=(label,value,known)=>'<div class="upgrade-check '+(known?'known':'unknown')+'"><span>'+esc(label)+'</span><b>'+esc(value)+'</b><i>'+(known?'已确认':'需核验')+'</i></div>';
    const protectionText=a.protection||a.boot?[a.protection?'保护机制':'',a.boot?'Bootloader':''].filter(Boolean).join(' · '):'未核验';
    const storageText=a.ecc||a.xip?[a.ecc?'Flash ECC':'',a.xip?'外部 Flash / XIP':''].filter(Boolean).join(' · '):'未核验';
    const body='<div class="upgrade-panel upgrade-'+a.level+'"><div class="upgrade-head"><div><span>固件升级适配</span><b>'+esc(a.label)+'</b><p>'+esc(a.note)+'</p></div><div class="upgrade-score"><b>'+a.score+'</b><span>/ 100</span></div></div><div class="upgrade-grid">'+check('片上 Flash',d.fl?memory(d.fl):'未核验',Boolean(d.fl))+check('Bank / RWW / Swap',a.bankText,a.dual||a.rww||a.swap)+check('编程 / 擦除粒度',a.eraseText,a.eraseText!=='未核验')+check('擦写寿命 / 保持',a.enduranceText,a.enduranceText!=='未核验')+check('保护 / 启动路径',protectionText,a.protection||a.boot)+check('ECC / 外部暂存',storageText,a.ecc||a.xip)+'</div><p class="upgrade-note">依据 '+a.items.length+' 条原厂 Flash 原文、'+a.sourceCount+' 份资料生成。分数只表示升级设计资料与基础条件的完备度，不代表厂商保证支持 OTA，也不能替代安全启动与断电测试。</p></div>';
    return detailAccordion('固件升级适配评估',body,false,a.label);
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
  function documentUrl(doc){
    if(!doc||typeof doc!=='object')return '';
    const direct=String(doc.url||doc.href||'').trim();
    if(/^https?:\/\//i.test(direct))return direct;
    const legacy=String(doc.name||'').trim();
    return /^https?:\/\//i.test(legacy)?legacy:'';
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
    const drawings=(Array.isArray(d.docs)?d.docs:[]).filter(doc=>doc.kind==='package_drawing'&&safeHttpUrl(documentUrl(doc)));
    const datasheets=(Array.isArray(d.docs)?d.docs:[]).filter(doc=>isDatasheetDocument(doc)&&safeHttpUrl(documentUrl(doc)));
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
    if(doc.path&&!documentUrl(doc))return {label:'Pack 内文件',className:'local'};
    return {label:documentUrl(doc)?'官方链接':'来源记录',className:'neutral'};
  }
  function documentRows(items){
    return `<div class="document-list">${items.map(doc=>{const url=safeHttpUrl(documentUrl(doc));const title=doc.title||doc.name||doc.path||'官方资料';const status=documentStatus(doc);const detail=[doc.version?`版本 ${doc.version}`:'',doc.path?`Pack 路径 ${doc.path}`:''].filter(Boolean).join(' · ');const body=`<span class="document-icon">${url?'↗':'DOC'}</span><span class="document-main"><b>${esc(title)}</b>${detail?`<small>${esc(detail)}</small>`:''}</span><span class="document-status ${status.className}">${esc(status.label)}</span>`;return url?`<a class="document-row" href="${esc(url)}" target="_blank" rel="noopener">${body}</a>`:`<div class="document-row document-local">${body}</div>`}).join('')}</div>`;
  }
  function documentsSection(d){
    const docs=Array.isArray(d.docs)?d.docs:[];
    const sourceUrls=safeHttpUrls(d.src);
    const manuals=docs.filter(isManualDocument);
    const sources=docs.filter(doc=>!isManualDocument(doc)&&doc.kind!=='package_drawing');
    sourceUrls.forEach((sourceUrl,index)=>{if(!sources.some(doc=>safeHttpUrl(documentUrl(doc))===sourceUrl)&&!manuals.some(doc=>safeHttpUrl(documentUrl(doc))===sourceUrl))sources.push({title:sourceUrls.length>1?`官方来源页面 ${index+1}`:'官方来源页面',url:sourceUrl,kind:'source'})});
    const manualBlock=manuals.length?documentRows(manuals):'<div class="feature-panel"><div class="feature-label">当前来源没有可直接打开的手册；Pack 内相对路径不会伪装成网页链接。</div></div>';
    const sourceBlock=sources.length?`<h3 class="document-group-title">产品页与器件包来源 · ${sources.length}</h3>${documentRows(sources)}`:'';
     return detailAccordion('官方手册',`${manualBlock}${sourceBlock}<p class="score-note">“已核验”表示链接审计可访问或来自厂商官方文档接口；“厂商限流/防护”不等于链接失效。</p>`,false,`${manuals.length} 份`);
  }
  function isUsableManualDocument(doc){
    if(!doc||!safeHttpUrl(documentUrl(doc)))return false;
    const status=String(doc.status||doc.verification_status||'').toLowerCase();
    if(status==='invalid'||status.includes('limited')||status.includes('rate')||status.includes('waf'))return false;
    return isManualDocument(doc)||/\.pdf(?:$|[?#])/i.test(documentUrl(doc));
  }
  function missingInfoReasons(d){
    const reasons=[];
    const docs=Array.isArray(d?.docs)?d.docs:[];
    const hasKnownNonNegative=(key)=>Object.prototype.hasOwnProperty.call(d||{},key)&&Number.isFinite(Number(d[key]))&&Number(d[key])>=0;
    if(!docs.some(isUsableManualDocument))reasons.push('没有可用手册');
    if(!packageNames(d).length)reasons.push('封装未核验');
    if(!String(d?.c||d?.a||'').trim())reasons.push('内核未核验');
    if(!(Number(d?.hz)>0))reasons.push('主频未核验');
    if(!hasKnownNonNegative('fl'))reasons.push('Flash 未核验');
    if(!hasKnownNonNegative('ra'))reasons.push('RAM 未核验');
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
    const configured=String((quoteProvider==='lcsc'?(readStoredString('mcul_lcsc_quote_api')||window.MCUS_LCSC_QUOTE_API):window.MCUS_QUOTE_API)||window.MCUS_QUOTE_API||'').trim();
    if(configured){try{const url=new URL(configured,location.href);if((url.protocol==='https:'||url.protocol==='http:')&&(!url.pathname||url.pathname==='/'))url.pathname='/api/quotes';return url.href}catch(_){return configured}}
    if(location.protocol==='https:'||location.protocol==='http:')return new URL('/api/quotes',location.href).href;
    return '';
  }
  function safeHttpUrl(value){try{const url=new URL(value);return url.protocol==='https:'||url.protocol==='http:'?url.href:''}catch(_){return ''}}
  function safeHttpUrls(value){return [...new Set(String(value||'').split(/[;\r\n]+/).map(item=>safeHttpUrl(item.trim())).filter(Boolean))]}
  function quoteMessage(code,fallback){
    const messages={feature_disabled:'实时询价暂未开放。',not_configured:quoteProvider==='lcsc'?'立创商城开放接口尚未配置。':'云汉芯城询价尚未配置。',invalid_part:'该订货号不适合直接询价。',invalid_quantity:'询价数量无效。',ickey_auth_error:'云汉芯城鉴权失败，请检查开放平台配置。',ickey_invalid_config:'云汉芯城接口配置无效。',ickey_api_error:'云汉芯城接口暂时不可用，请稍后重试。',lcsc_auth_error:'立创商城接口鉴权失败，请检查开放平台配置。',lcsc_invalid_config:'立创商城接口配置无效。',lcsc_api_error:'立创商城接口暂时不可用，请稍后重试。',no_strict_matches:'没有找到与完整订货号精确匹配的可售货源。',network_error:'无法连接询价服务，请检查网络后重试。'};
    return messages[code]||fallback||'询价失败，请稍后重试。';
  }
  function quotePanelHtml(part,data){
    const quotes=Array.isArray(data.quotes)?data.quotes:[];
    const quantity=Math.max(1,Number(data.quantity)||1);
    const updated=data.updatedAt?new Date(data.updatedAt).toLocaleString('zh-CN',{hour12:false}):'刚刚';
    const providerName=String(data.providerName|| (quoteProvider==='lcsc'?'立创商城':'云汉芯城'));
    return `<div class="quote-head"><div><b>${esc(providerName)}询价 · ${esc(part)}</b><br><span>${quotes.length} 条精确货源 · 按 ${quantity} 件 · ${esc(updated)}</span></div></div><form class="quote-quantity" data-quote-quantity><label>询价数量</label><input name="quantity" type="number" min="1" max="1000000" step="1" value="${quantity}" inputmode="numeric"><button type="submit">更新</button></form>${quotes.length?`<div class="quote-list">${quotes.map(item=>{const link=safeHttpUrl(item.url);const price=Number(item.price);const stock=Number(item.stock);const moq=Number(item.moq);const tiers=Array.isArray(item.priceTiers)?item.priceTiers:[];return `<div class="quote-row"><div><span class="quote-shop">${esc(item.shop||providerName)}</span><p class="quote-title">${esc(item.title||part)}</p><p class="quote-meta">库存 ${Number.isFinite(stock)?stock.toLocaleString('zh-CN'):'—'} · MOQ ${Number.isFinite(moq)?moq:'—'}${item.package?` · ${esc(item.package)}`:''}${item.dateCode?` · 批次 ${esc(item.dateCode)}`:''}${item.leadTime?` · ${esc(item.leadTime)}`:''}</p>${tiers.length?`<div class="quote-tiers">${tiers.slice(0,4).map(tier=>`<span>${esc(tier.quantity)}+ ¥${Number(tier.price).toFixed(4)}</span>`).join('')}</div>`:''}</div><div class="quote-price"><b>¥${Number.isFinite(price)?price.toFixed(4):'—'}</b><small>/ 件</small>${link?`<a href="${esc(link)}">查看货源 ↗</a>`:''}</div></div>`}).join('')}</div>`:`<div class="quote-status">${esc(quoteMessage('no_strict_matches'))}</div>`}<p class="quote-note">数据来自${esc(providerName)}开放接口，按完整订货号精确匹配；库存、批次、交期和最终含税成交价以供应商结算页为准。</p>`;
  }
  async function requestQuotes(part,quantity=1,targetPanel=null){
    const panel=targetPanel||$('#quote-panel');if(!panel)return;
    part=String(part||'').trim().toUpperCase();
    quantity=Math.max(1,Math.min(1000000,Math.floor(Number(quantity)||1)));
    if(!/^[A-Z0-9][A-Z0-9+._\/-]{3,63}$/.test(part)){panel.hidden=false;panel.innerHTML=`<div class="quote-head"><b>实时询价</b></div><div class="quote-status">${esc(quoteMessage('invalid_part'))}</div>`;return}
    document.querySelectorAll('[data-quote-part]').forEach(button=>button.classList.toggle('active',button.dataset.quotePart===part));
    const providerName=quoteProvider==='lcsc'?'立创商城':'云汉芯城';
    panel.hidden=false;panel.dataset.part=part;panel.dataset.quantity=String(quantity);panel.innerHTML=`<div class="quote-head"><b>${providerName}询价 · ${esc(part)}</b></div><div class="quote-status">正在查询精确型号的实时库存与阶梯价…</div>`;
    panel.scrollIntoView({behavior:'smooth',block:'nearest'});
    const endpoint=quoteEndpoint();
    if(!endpoint){panel.innerHTML=`<div class="quote-head"><b>${providerName}询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage('not_configured'))}</div>`;return}
    if(quoteAbort)quoteAbort.abort();quoteAbort=new AbortController();
    try{
      const url=new URL(endpoint,location.href);url.searchParams.set('part',part);url.searchParams.set('quantity',String(quantity));url.searchParams.set('provider',quoteProvider);
      const response=await fetch(url.href,{headers:{accept:'application/json'},signal:quoteAbort.signal});
      const data=await response.json().catch(()=>({}));
      if(panel.dataset.part!==part)return;
      if(!response.ok)throw Object.assign(new Error(data.message||''),{code:data.code||'ickey_api_error'});
      if(!Array.isArray(data.quotes))throw Object.assign(new Error('立创商城接口尚未接入。'),{code:'not_configured'});
      panel.innerHTML=quotePanelHtml(part,data);
      const quantityForm=panel.querySelector('[data-quote-quantity]');if(quantityForm)quantityForm.onsubmit=event=>{event.preventDefault();requestQuotes(part,quantityForm.elements.quantity.value,panel)};
    }catch(error){
      if(error&&error.name==='AbortError')return;
      if(panel.dataset.part!==part)return;
      const code=error&&error.code?error.code:'network_error';
      panel.innerHTML=`<div class="quote-head"><b>${providerName}询价 · ${esc(part)}</b></div><div class="quote-status">${esc(quoteMessage(code,error&&error.message))}<br><button class="quote-retry" data-quote-retry="${esc(part)}">重新询价</button></div>`;
      const retry=panel.querySelector('[data-quote-retry]');if(retry)retry.onclick=()=>requestQuotes(retry.dataset.quoteRetry,panel.dataset.quantity,panel);
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
    const coreBody=`<div class="spec-grid">${spec(engineeringValue(d.c||d.a),'处理器核心')}${spec(d.cc?d.cc+' core':'未核验','核心数')}${spec(d.hz?clock(d.hz):'未核验','最高核心频率')}${spec(voltageRange(d)==='—'?'未核验':voltageRange(d),'工作电压范围')}${spec(engineeringMemory(d.fl),'片上 Flash')}${spec(engineeringMemory(d.ra),'片上 RAM')}${spec(yesNoValue(d.fpu),'FPU')}${spec(yesNoValue(d.dsp),'DSP')}${spec(yesNoValue(d.mpu),'MPU')}${spec(yesNoValue(d.tz),'TrustZone')}${spec(Array.isArray(d.eb)?(facts.externalBus.length?facts.externalBus.join(' / '):'无'):'未核验','外部存储总线')}</div>`;
    const peripheralBody=`<div class="spec-grid">${spec(facts.timerWidths.length?facts.timerWidths.join(' / ')+' bit':'未核验','定时器位宽')}${spec(engineeringValue(d.tim),'TIM 总数')}${spec(engineeringValue(d.adcu),'ADC 转换器单元')}${spec(engineeringValue(d.adch),'ADC 通道（不含引脚）')}${spec(facts.adcResolution?facts.adcResolution+' bit':'未核验','ADC 分辨率')}${spec(facts.adcRate?clock(facts.adcRate):'未核验','ADC 采样率')}${spec(engineeringValue(d.dac),'DAC 数量（源表口径）')}${spec(facts.dacResolution?facts.dacResolution+' bit':'未核验','DAC 分辨率')}${spec(facts.dacRate?clock(facts.dacRate):'未核验','DAC 速度')}${spec(engineeringValue(d.gpio),'GPIO')}${spec(engineeringValue(d.dma),'DMA 通道')}${spec(engineeringValue(d.sercom),'SERCOM / FLEXCOM')}${spec(engineeringValue(d.spi),'SPI')}${spec(engineeringValue(d.i2c),'I²C')}${spec(engineeringValue(d.usart),'USART')}${spec(engineeringValue(d.uart),'UART')}${spec(engineeringValue(d.can),'CAN')}${spec(engineeringValue(d.usbd),'USB Device')}${spec(engineeringValue(d.usbh),'USB Host')}${spec(engineeringValue(d.eth),'Ethernet')}${spec(engineeringValue(d.pin),'封装引脚数')}</div>`;
    const featureBody=`<div class="feature-panel">${accelerators.length?`<div class="feature-label">已确认的加速器与厂商能力 · 点击查看说明</div>${acceleratorGrid(accelerators)}`:'<div class="feature-label">当前来源没有已确认的专用加速器或能力记录</div>'}${(d.pending||[]).length?`<div class="feature-label feature-label-spaced">待逐器件官方文档核验</div><div class="feature-list">${d.pending.map(x=>{const label=featureLabel(x);return `<span class="feature-chip pending" title="${esc(label)}">${esc(label)}</span>`}).join('')}</div><p class="feature-note">候选项不代表该具体后缀型号已经确认支持。</p>`:''}</div>`;
    const scoreBody=`<div class="score-panel">${[['计算',d.cs],['存储',d.ms],['外设',d.ps],['加速器',d.acs]].map(x=>`<div class="score-row"><label>${x[0]}</label><div class="score-bar"><i style="width:${Math.max(0,Math.min(100,x[1]||0))}%"></i></div><b>${value(x[1])}</b></div>`).join('')}<p class="score-note">数据覆盖率 ${value(d.cov)}%。这是选型排序指标，不是实测性能。CoreMark：${value(d.cm)} · DMIPS：${value(d.dm)}。</p></div>`;
    const quoteActionLabel=quoteProvider==='lcsc'?'立创查询':'询价';
    const partsBody=parts.length?`<div class="parts">${parts.map(p=>`<div class="part-row"><div><b>${esc(p.n)}</b><p>后缀 ${esc(p.s||'—')} · 封装码 ${esc(p.p||'—')} · 温度码 ${esc(p.t||'—')} · 包装 ${esc(p.k||'—')}</p></div><div class="part-actions"><span class="verified">✓ 已核验</span><button class="quote-trigger" data-quote-part="${esc(p.n)}">${quoteActionLabel}</button></div></div>`).join('')}</div>`:'<div class="feature-panel"><div class="feature-label">完整订货号尚未导入；不会根据后缀组合自动生成。请明确输入厂商完整订货号后询价。</div><form class="quote-manual" id="quote-manual"><input id="quote-manual-part" autocomplete="off" autocapitalize="characters" spellcheck="false" placeholder="输入完整订货号，如 STM32F429ZIT6"><button type="submit">询价</button></form></div>';
    const warningTone=missingInfoLevel(missingReasons)==='manual'?' manual-warning':'';
    layer.innerHTML=`<div class="detail-backdrop"></div><section class="detail-page"><div class="detail-header ${missingInfoHeaderClass(missingReasons)}"><button class="back-btn">‹</button><div class="detail-title"><h1>${missingInfoBadge(d)}${esc(d.n)}</h1><p>${esc(d.m)} · ${esc(d.s)} · ${esc(productType(d.pt))}</p></div><div class="detail-score"><b>${value(d.idx)}</b><span>选型指数 / 100</span></div></div><div class="detail-hero"><div class="detail-path">${esc(d.m)} › ${esc(d.s)} › ${esc(d.l)} › ${esc(productType(d.pt))}</div><div class="detail-model"><b>${esc(d.n)}</b><span>变体码 ${esc(d.v||'—')}</span></div>${missingReasons.length?`<div class="data-warning-note${warningTone}">⚠ 关键资料缺失：${esc(missingReasons.join('、'))}</div>`:''}${releaseFacts(d)}</div>${detailAccordion('核心与存储',coreBody,true,d.c||d.a||'核心')} ${powerSection(d)}${detailAccordion('外设资源',peripheralBody,true,`${[d.tim,d.adcu,d.adch,d.gpio,d.uart,d.usart].filter(v=>v!==undefined&&v!==null&&v!=='').length} 类已知`)}${packageSection(d)}${documentsSection(d)}${inventorySection(d)}${detailAccordion('厂商加速器与特性',featureBody,false,features.length?`${features.length} 项`:'未确认')}${detailAccordion('评分拆解',scoreBody,false,d.idx?`${d.idx} / 100`:'未评分')}${detailAccordion('官方完整订货号',`${partsBody}<div class="quote-panel" id="quote-panel" hidden aria-live="polite"></div>`,false,`${parts.length} 个`)}<button class="detail-action primary" id="detail-compare">${selected?'✓ 已加入对比':'＋ 加入参数对比'}</button>${sourceActions}</section>`;
    if(!quotesEnabled){const manual=layer.querySelector('.quote-manual');if(manual){const label=manual.parentElement.querySelector('.feature-label');if(label)label.textContent='完整订货号尚未导入；不会根据后缀组合自动生成。'}layer.querySelectorAll('.quote-trigger,.quote-manual,.quote-panel').forEach(element=>element.remove())}
    const coreSection=layer.querySelector('.detail-section');
    if(coreSection)coreSection.insertAdjacentHTML('afterend',memoryEvidenceSection(d)+flashUpgradeSection(d));
    const detailSpecGrids=layer.querySelectorAll('.spec-grid');
    const coreCountValue=detailSpecGrids[0]?.querySelector('.spec-cell:nth-child(2) b');
    if(coreCountValue)coreCountValue.textContent=d.cc?d.cc+' core':'—';
    if(detailSpecGrids[1])detailSpecGrids[1].insertAdjacentHTML('beforeend',spec(value(d.usb),'USB（角色未标明）'));
    if(d.m==='Microchip'){
      layer.querySelector('.detail-title p').textContent=`${vendorName(d.m)} · ${d.s}`;
      const path=layer.querySelector('.detail-path');path.textContent=path.textContent.replace(/^Microchip/,vendorName(d.m));
    }
    if((d.boards||[]).length)layer.querySelector('.detail-hero').insertAdjacentHTML('beforeend',boardTags(d,true));
    const setupSpecExpanders=()=>{
      layer.querySelectorAll('[data-spec-expand]').forEach(button=>{
        const cell=button.closest('.spec-cell'),specValue=cell?.querySelector('.spec-value');
        if(!cell||!specValue)return;
        if(cell.classList.contains('expanded'))return;
        const overflows=specValue.scrollWidth>specValue.clientWidth+1;
        button.hidden=!overflows;
        if(!overflows)return;
        button.onclick=()=>{
          const expanded=cell.classList.toggle('expanded');
          button.textContent=expanded?'收起':'展开';
          button.setAttribute('aria-expanded',String(expanded));
        };
      });
    };
    requestAnimationFrame(()=>{
      layer.classList.add('open');
      setupSpecExpanders();
      requestAnimationFrame(setupSpecExpanders);
      window.setTimeout(()=>{if(state.detail===d.id)setupSpecExpanders()},240);
    });
    layer.querySelector('.back-btn').onclick=closeDetail;
    layer.querySelector('.detail-backdrop').onclick=closeDetail;
    $('#detail-compare').onclick=()=>{toggleCompare(d.id);openDetail(d.id)};
    layer.querySelectorAll('[data-accelerator-toggle]').forEach(button=>button.onclick=()=>{
      const card=button.closest('[data-accelerator-card]'),expanded=card.classList.toggle('active');
      button.setAttribute('aria-expanded',String(expanded));
      const explanation=card.querySelector('[data-accelerator-explanation]');
      if(explanation)explanation.hidden=!expanded;
      if(expanded)layer.querySelectorAll('[data-accelerator-card].active').forEach(other=>{
        if(other===card)return;
        other.classList.remove('active');
        const toggle=other.querySelector('[data-accelerator-toggle]'),body=other.querySelector('[data-accelerator-explanation]');
        if(toggle)toggle.setAttribute('aria-expanded','false');
        if(body)body.hidden=true;
      });
    });
    layer.querySelectorAll('[data-quote-part]').forEach(button=>button.onclick=()=>requestQuotes(button.dataset.quotePart));
    const manualForm=layer.querySelector('#quote-manual');
    if(manualForm)manualForm.onsubmit=event=>{event.preventDefault();requestQuotes(layer.querySelector('#quote-manual-part').value)};
  }
  function closeDetail(){state.detail=null;if(quoteAbort){quoteAbort.abort();quoteAbort=null}const layer=$('#detail-layer');layer.classList.remove('open');setTimeout(()=>{if(!state.detail)layer.innerHTML=''},220)}
  function render(){renderHeader();if(state.tab==='catalog')renderCatalog();else if(state.tab==='search')renderSearch(false);else if(state.tab==='compare')renderCompare();else if(state.tab==='data')renderData();else renderAuthor();updateNav()}
  window.MCUL={handleAndroidBack:function(){if(state.detail){closeDetail();return true}if(state.tab==='catalog'){if(state.browse.line){state.browse.line=null;renderCatalog();return true}if(state.browse.series){state.browse.series=null;renderCatalog();return true}if(state.browse.vendor){state.browse.vendor=null;renderCatalog();return true}}if(state.tab!=='catalog'){setTab('catalog');return true}return false}};
  document.querySelectorAll('#bottom-nav button').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
  $('#snapshot-pill').textContent=`${count(catalog.meta.devices)} DEVICES · OFFLINE`;
  $('#splash').remove();$('#app').hidden=false;render();if(previewDevice&&byId.has(previewDevice))setTimeout(()=>openDetail(previewDevice),50);
})();
