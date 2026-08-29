/* MCUS on-device slot model. It is intentionally compact and deterministic:
 * weighted vocabulary understands MCU language, while catalog fields enforce
 * the actual electrical/resource constraints. No prompt leaves the device.
 */
window.MCUS_LOCAL_MODEL = {
  name: "MCUS MCU 选型槽位模型",
  version: "0.7.0",
  type: "natural-language-slot-constraint",
  vendors: [
    {label:"STMicroelectronics", terms:["stmicroelectronics","stm32","意法"]},
    {label:"Espressif", terms:["espressif","esp32","esp8266","乐鑫"]},
    {label:"Qinheng", terms:["qinheng","wch","ch32","沁恒","青稞"]},
    {label:"HPMicro", terms:["hpmicro","hpm","先楫"]},
    {label:"Microchip", terms:["microchip","atmel","avr","samd","pic"]},
    {label:"STC", terms:["stc","宏晶"]},
    {label:"GigaDevice", terms:["gigadevice","gd32","兆易"]},
    {label:"MindMotion", terms:["mindmotion","mm32","灵动微"]},
    {label:"Nuvoton", terms:["nuvoton","numicro","新唐"]},
    {label:"Puya", terms:["puya","py32","普冉"]},
    {label:"Geehy", terms:["geehy","apm32","极海"]},
    {label:"Infineon", terms:["infineon","psoc","xmc","英飞凌"]},
    {label:"Texas Instruments", terms:["texas instruments","ti芯片","mspm","msp430","德州仪器"]},
    {label:"Renesas", terms:["renesas","瑞萨","瑞萨电子","ra系列","rx系列","rl78","rh850","synergy"]},
    {label:"Artery", terms:["artery","arterytek","雅特力","at32","at32f","at32a","at32l","at32m","at32wb"]},
    {label:"Allwinner", terms:["allwinner","xradio","xr806","全志"]},
    {label:"MicroPy MCU", terms:["micropython","micro python","micropy","canmv","rp2040","rp2350","rp2354","k210","k230","k510","kendryte","树莓派"]}
  ],
  cores: [
    {label:"Cortex-M0", terms:["cortex-m0","cortex m0","arm m0","m0"]},
    {label:"Cortex-M0+", terms:["cortex-m0+","cortex m0+","arm m0+","m0+"]},
    {label:"Cortex-M3", terms:["cortex-m3","cortex m3","arm m3","m3"]},
    {label:"Cortex-M4", terms:["cortex-m4","cortex m4","arm m4","m4"]},
    {label:"Cortex-M7", terms:["cortex-m7","cortex m7","arm m7","m7"]},
    {label:"Cortex-M23", terms:["cortex-m23","cortex m23","arm m23","m23"]},
    {label:"Cortex-M33", terms:["cortex-m33","cortex m33","arm m33","m33"]},
    {label:"Cortex-M55", terms:["cortex-m55","cortex m55","arm m55","m55"]},
    {label:"RISC-V", terms:["risc-v","risc v","riscv","青稞","c908","hazard3"]},
    {label:"8051", terms:["8051","mcs-51","mcs51"]}
  ],
  features: [
    {key:"serial", terms:["uart","usart","串口","串行"]},
    {key:"spi", terms:["spi","串行外设接口"]},
    {key:"i2c", terms:["i2c","i²c","两线总线"]},
    {key:"i2s", terms:["i2s","i²s","音频接口"]},
    {key:"can", terms:["can-fd","canfd","can总线","twai","can"]},
    {key:"usbAny", terms:["usb","usb设备","usb主机","otg"]},
    {key:"eth", terms:["ethernet","以太网"]},
    {key:"wifi", terms:["wi-fi","wifi","无线局域网"]},
    {key:"bluetooth", terms:["bluetooth","蓝牙","ble"]},
    {key:"cam", terms:["camera","摄像头","相机","dvp","dcmi"]},
    {key:"display", terms:["display","lcd","显示屏","显示接口"]},
    {key:"pwm", terms:["pwm","脉宽","电机控制"]},
    {key:"adch", terms:["adc","模拟通道","模数转换"]},
    {key:"gpio", terms:["gpio","通用io","通用 i/o"]},
    {key:"tim", terms:["timer","定时器","计数器"]}
  ],
  aliases: {
    serial: ["串行口","通信口","通讯口","调试口","调试串口","异步串口","多串口","串行接口","几路串口","两三路串口","一两个串口"],
    timer: ["计时器","定时资源","定时器资源","高级定时器"],
    memory: ["运行内存","片上内存","片上ram","内存容量"],
    storage: ["程序存储","程序空间","代码空间","闪存容量"],
    wireless: ["无线网络","无线连接","联网","无线上网"],
    display: ["屏幕","显示屏","接屏","带屏","带显示","带屏的小设备","液晶","人机界面","图形界面"],
    camera: ["图像采集","图像传感器","接摄像头","需要接摄像头","视觉","摄像头接口"],
    audio: ["音频","麦克风","数字音频"],
    motor: ["电机","电机控制板","电机驱动","马达","伺服","无刷","步进","逆变","运动控制","FOC"],
    lowPower: ["低功耗","省电","节能","吃电少","耗电少","省电一点","功耗别太高","电池供电","电池应用","电池撑得久","续航长","续航久","待机时间长"],
    compact: ["小封装","小尺寸","封装小一点","少引脚","节省板面积","空间紧张"],
    highPerformance: ["高性能","高速","跑得快","跑得动","快一点","处理得快","算力","实时性","响应快","反应快","频率高一点","高频一点","性能别太差"],
    domestic: ["国产","国产替代","国内厂商"]
  },
  profiles: [
    {key:"motor_control", label:"电机控制", terms:["电机","电机控制板","电机驱动","马达","伺服","无刷","步进","逆变","运动控制","foc"], softMinimums:{pwm:1,tim:1}},
    {key:"low_power", label:"低功耗", terms:["低功耗","省电","节能","吃电少","耗电少","电池供电","电池供电的传感器","电池应用","便携传感器","电池撑得久","续航长","续航久","待机时间长"], preferences:["lowPower"]},
    {key:"wireless", label:"无线连接", terms:["wifi","无线网络","无线连接","联网","无线上网"], preferences:["wireless"]},
    {key:"display", label:"显示界面", terms:["display","显示界面","屏幕","显示屏","接屏","带屏","带屏的小设备","液晶","人机界面","图形界面"], softMinimums:{display:1}},
    {key:"camera", label:"图像采集", terms:["camera","摄像头","接摄像头","需要接摄像头","图像采集","图像传感器","视觉"], softMinimums:{cam:1}},
    {key:"audio", label:"音频处理", terms:["音频","麦克风","数字音频"], softMinimums:{i2s:1}},
    {key:"industrial", label:"工业通信", terms:["工业通信","工业控制","现场总线","伺服驱动"], softMinimums:{can:1}},
    {key:"sensor", label:"传感器采集", terms:["传感器","电池供电的传感器","便携传感器","数据采集","模拟采集"], softMinimums:{adch:1}},
    {key:"compact", label:"小型化设计", terms:["小封装","小尺寸","少引脚","节省板面积","空间紧张"], preferences:["compact"]},
    {key:"domestic", label:"国产替代", terms:["国产","国产替代","国内厂商"], preferences:["domestic"]}
  ],
  preferences: [
    {key:"lowPower", label:"优先低功耗", terms:["低功耗","省电","节能","电池供电","待机时间长"]},
    {key:"highPerformance", label:"优先高性能", terms:["高性能","高速","算力","实时性","响应快"]},
    {key:"compact", label:"优先小封装", terms:["小封装","小尺寸","少引脚","节省板面积","空间紧张"]},
    {key:"ecosystem", label:"优先生态成熟", terms:["生态成熟","资料多","资料全","文档多","开发方便","好开发","容易开发","开发简单","好上手","上手快","工具链成熟","社区多","arduino"]},
    {key:"domestic", label:"优先国产", terms:["国产","国产替代","国内厂商"]},
    {key:"largeMemory", label:"优先大容量", terms:["大内存","内存大","内存别太小","内存够用","ram大一点","大容量","存储大"]},
    {key:"morePeripherals", label:"优先外设丰富", terms:["外设丰富","外设多一点","接口多","接口多一点","多几个接口","串口多","多路接口"]}
  ]
};
