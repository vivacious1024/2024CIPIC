<template>
  <div id="guide" style="margin: 0 auto;" v-if="guide">
    <div style="text-align: center;">
      <n-gradient-text type="info">
        <br>
        <img src="bigfive.jpg" style="height: 30rem" />
        <!-- <h1>大五人格测试</h1> -->
      </n-gradient-text>
      
      <!-- <n-timeline horizontal style="margin-top: 1rem; justify-content: center;">
        <n-timeline-item
          type="info"
          title="采集数据"
          content="佩戴设备并连接"
        />
        <n-timeline-item
          title="分析数据"
          content="服务器进行分析"
        />
        <n-timeline-item
          title="得出结果"
          content="查看分析报告"
        />
      </n-timeline> -->
      
      <!-- <n-alert title="什么是大五人格？" type="info" style="width: 600px; margin: 0 auto;">
        五大性格特质（Big Five personality traits），又叫「五因素模型」或「大五人格模型」，是目前公认比较全面的人格分析模型，通过大五人格量表（The Big Five）进行测量。近年来，研究者们在人格描述模式上形成了比较一致的共识，提出了人格的大五模式， Goldberg（1992）把它称之为人格心理学中的一场革命，研究者通过词汇学的方法，发现大约有五种特质可以涵盖人格描述的所有方面。 大五人格（OCEAN），也被称之为人格的海洋，它们分别是：

        <li>开放性（Openness）：指个体对经验持开放、探求的态度。</li>
        <li>尽责性（Conscientiousness）：指个体在目标导向行为上的组织、坚持和动机。</li>
        <li>外向性（Extraversion）：指个体对外部世界的积极投入程度。</li>
        <li>宜人性（Agreeableness）：指个体在合作与社会和谐性方面的差异。</li>
        <li>神经质（Neuroticism）：指个体体验消极情绪的倾向。</li>
      </n-alert> -->
    </div>

    <div style="text-align: center;">
      <p style="color: #ffffff; font-size: 4rem;">欢迎您进行大五人格测评！</p>
      <p>
        <n-button class="button" color="#8a2be2" style="font-size: 2em;" @click="connect">{{this.stage}}{{this.id}}{{this.token}}连接设备</n-button>
        <n-button class="button" color="#8a2be2" style="font-size: 2em;" @click="start">开始采集</n-button>
      </p>
      <p style="color: #ffffff;">{{ (connected)?'连接成功'+this.id:(stage>=1)?'连接中':'未连接' }}</p>
    </div>
  </div>

  <div id="show" v-if="show" class="content">
    <!-- <p class="word">{{ ctx[0] }}</p> -->
    <p class="word" v-if="!ctx3">{{ ctx[1] }}</p>
    <p class="word" v-if="!ctx3">{{ ctx[2] }}</p>
    <p class="word" :style="ctx3">{{ ctx[3] }}</p>
    <p class="word" v-if="!ctx3">{{ ctx[4] }}</p>
    <p class="word" v-if="!ctx3">{{ ctx[5] }}</p>
    <!-- <p class="word">{{ ctx[6] }}</p> -->
    <videoPlay v-if="video_playing" v-bind="video_options" @play="onVideoPlay" @ended="onVideoEnded" @pause="onVideoPause"></videoPlay>
  </div>
</template>


<script>
import screenfull from 'screenfull';

import "vue3-video-play/dist/style.css";
import { videoPlay } from 'vue3-video-play';
import { reactive } from 'vue';

export default {
  data() {
    let token = '';//令牌

    let port = {};//串口
    let recvBuffer = [];//接收缓冲区
    let stage = 0;
    let connected = false;
    let blueMsg = '';
    /**
     * stage 0: 未连接
     * stage 1: 开始并发送 AT+ROLE=0\r\n 等待
     * stage 2: 接收到 +IM_MASTER
     * stage 3: 继续发送 AT+SCAN=1,5 等待
     * stage 4: 接收到扫描数据，等待选择
     * stage 5: 发送 AT+CONN=0,A6C080030009 等待
     */

    let guide = true;
    let show = false;
    let id = '';

    let ctx = ['　', '　', '　', '　', '　', '　', '　'];//显示内容

    let eegdata = [];
    let eegdata_pkg = 0;
    let eegdata_step = 0;
    let post_pid = 0;
    let eegdata_post = [];

    let video_options = {};
    let video_playing = false;//视频播放中
    let video_resolve = {};

    let ctx3 = "";

    return { token, port, recvBuffer, stage, guide, show, ctx, id, eegdata, eegdata_pkg, eegdata_step, post_pid, eegdata_post, connected, video_options, video_playing, video_resolve, ctx3, blueMsg };
  },
  methods: {
    async buf2hex(buffer) {//转换为16进制字符串
      return Array.prototype.map.call(new Uint8Array(buffer), x => ('00' + x.toString(16)).slice(-2)).join('');
    },
    async hex2buf(hex) {//转为16进制数组
      return new Uint8Array(hex.match(/[\da-f]{2}/gi).map(function (h) {
        return parseInt(h, 16)
      }))?.buffer;
    },
    async getToken() {
      const res = await fetch('http://localhost:3000/getToken');
      const result = await res.json();
      return result.token;
    },
    async logEvent(event) {
      const res = await fetch(`http://localhost:3000/event?token=${this.token}`, {
        method: 'POST',
        body: JSON.stringify({ event }),
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const result = await res.json();
      return result;
    },
    async saveEEG(pid, data, end = false) {
      const res = await fetch(`http://localhost:3000/saveEEG?token=${this.token}&pid=${pid}${end?'&end=true':''}`, {
        method: 'POST',
        body: JSON.stringify({ data: await this.buf2hex(data) }),
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const result = await res.json();
      return result;
    },
    str2ascii(str) {//字符串转Unicode码点
      const ret = [];
      for(const i in str) {
        ret.push(str.codePointAt(i));
      }
      return ret;
    },
    recv2str(recv) {
      const str = [];
      for(const i of recv) {
        str.push(String.fromCodePoint(i));
      }
      return str.join('');
    },
    shuffle(arr) {//洗牌算法
      if (!arr || !arr.length) return -1;
      let length = arr.length;
      for (let i = length - 1; i > 0; i--) {
        let randomIndex = Math.floor((Math.random() * i)); // 生成一个随机数作为数组的一个index
        [arr[i], arr[randomIndex]] = [arr[randomIndex], arr[i]]; // 交换当前index的元素和生成的随机index的元素的值 
      }
      return arr;
    },
    async sendString(str) {
      if(this.port.writable) {
        const writer = this.port.writable.getWriter();
        const data = new Uint8Array(this.str2ascii(str));
        await writer.write(data);
        console.log(str, data);
        await writer.releaseLock();
      }
    },
    async sendHex(arr) {
      if(this.port.writable) {
        console.log(` winwin！`);
        const writer = this.port.writable.getWriter();
        const data = new Uint8Array(arr);
        await writer.write(data);
        console.log(data);
        await writer.releaseLock();
      }
    },
    async sleep(seconds) {
      return new Promise((resolve) => setTimeout(resolve, seconds));
    },
    async setLine(line, ctx) {
      this.logEvent(`${this.eegdata_pkg}`);
      this.ctx[line] = ctx;
    },
    async clearLine(line) {
      this.ctx[line] = '　';
    },
    async clearAll() {
      this.ctx = ['　', '　', '　', '　', '　', '　', '　'];
    },
    async waitKey(code) {//等待按键
      const startTime = Date.now();
      return new Promise((resolve, reject) => {
        const keyDown = (event) => {
          if(code === event.code || code?.includes(event.code)) {
            window.removeEventListener("keydown", keyDown, true);
            const endTime = Date.now();
            this.logEvent(`${this.eegdata_pkg},resolveKey=${event.code},${endTime - startTime}`);
            resolve(endTime - startTime);
          }
        }
        window.addEventListener("keydown", keyDown, true);
      });
    },
    async playMusic(name) {
      const audio = new Audio(`/music/${name}.mp3`);
      await audio.play();
      this.logEvent(`${this.eegdata_pkg},playMusic=${name}`);
    },
    async playVideo(name) {
      this.video_playing = true;
      
      const options = reactive({
        width: "800px", //播放器高度
        height: "450px", //播放器高度
        color: "#409eff", //主题色
        title: "", //视频名称
        src: `/video/${name}.mp4`, //视频源
        muted: false, //静音
        webFullScreen: true,
        speedRate: ["0.75", "1.0", "1.25", "1.5", "2.0"], //播放倍速
        autoPlay: false, //自动播放
        loop: false, //循环播放
        mirror: false, //镜像画面
        ligthOff: false, //关灯模式
        volume: 1.0, //默认音量大小
        control: false, //是否显示控制
        controlBtns: [
          "audioTrack",
          "quality",
          "speedRate",
          "volume",
          "setting",
          "pip",
          "pageFullScreen",
          "fullScreen",
        ],
      });
      this.video_options = options;
      this.logEvent(`${this.eegdata_pkg},playVideo=${name}`);
      return new Promise((resolve, reject) => {
        this.video_resolve = resolve;
      });
    },
    onVideoPlay() {
      this.logEvent(`${this.eegdata_pkg},videoPlay`);
    },
    onVideoPause() {
      this.logEvent(`${this.eegdata_pkg},videoPause`);
    },
    onVideoEnded() {
      this.video_playing = false;
      this.video_resolve();
      this.logEvent(`${this.eegdata_pkg},videoEnded`);
    },
    async execute() {
      // await this.ywt_workflow();
      // await this.fmz_video_workflow();
      await this.fmz_word_workflow();
    },
    async fmz_video_workflow() {
      await this.sleep(1000);
      await this.setLine(1, '指导语')
      await this.setLine(2, '欢迎参加本次实验');
      await this.setLine(3, '观看视频，在十字出现并消失后按空格开始播放视频');
      await this.setLine(4, '请尽量避免眨眼，吞咽口水等动作');
      await this.setLine(5, '如果理解指导语，按空格开始实验');
      await this.waitKey('Space');
      await this.logEvent(`${this.eegdata_pkg},start`); // 正式开始采集
      await this.clearAll();
      
      const seq = this.shuffle([ 1, 3, 4, 5, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]);

      for(const i of seq) {
        await this.setLine(3, '+');
        await this.sleep(5000);
        await this.playVideo(`Clip_${i}`);
        await this.setLine(2, '请选择看完视频的情感体验（小键盘输入0~7）');
        await this.setLine(3, '0: 生气  1: 反感  2: 害怕  3: 悲伤');
        await this.setLine(4, '4: 愉悦  5: 启发  6: 满足  7: 温柔');
        await this.waitKey([ 'Numpad0', 'Numpad1', 'Numpad2', 'Numpad3', 'Numpad4', 'Numpad5', 'Numpad6', 'Numpad7' ]);
        await this.clearAll();

        // 算式
        const p = ~~(Math.random()*100);
        const q = ~~(Math.random()*100);
        let r;
        if(Math.random() > 0.5) {
          r = p + q;
        } else {
          r = ~~(Math.random()*100);
        }
        await this.setLine(2, '请判断下面的等式是否成立');
        await this.setLine(3, `${p}+${q}=${r}`);
        this.logEvent(`${this.eegdata_pkg},${p}+${q}=${r},expect=${(p+q==r)?1:0}`);
        await this.setLine(4, '如果成立请按1，如果不成立请按0');
        await this.waitKey([ 'Numpad0', 'Numpad1' ]);
        await this.clearAll();
      }
      await this.logEvent(`${this.eegdata_pkg},stop`); // 试验结束
      await this.setLine(3, '实验结束，好耶');
      await this.stop();
    },
    id2class(id) { // [1,200]
      if (1 <= id && id <= 60) {
        return 1;
      } else if (61 <= id && id <= 120) {
        return 2;
      } else if (121 <= id && id <= 180) {
        return 3;
      } else {
        return 4;
      }
    },
    async fmz_word_workflow() {
      await this.sleep(1000);
      let seq = [];
      for(let i = 1; i<=200; ++i) {
        seq.push(i);
      }
      seq = this.shuffle(seq);
      const wordList = ['深情','真心','宝石','母亲','朋友','信心','才华','友谊','鲜花','美德','情侣','成就','天堂','美女','英雄','奖金','笑容','智慧','冠军','爱情','赞叹','信赖','体贴','拥抱','尊重','称赞','鼓励','爱慕','信任','欢呼','求婚','惊喜','庆祝','表扬','敬佩','奖励','祝福','微笑','自信','恋爱','欢快','永恒','乐观','满意','杰出','伟大','辉煌','欢喜','聪明','优异','崇高','高贵','高尚','欢乐','愉快','优秀','漂亮','浪漫','美好','美丽','成年','地位','首相','银行','规律','睡眠','皇帝','人情','军人','体操','权力','人民','眼睛','都市','形象','时机','首领','光线','关键','货币','爆发','包围','反攻','解除','抵御','旁观','出兵','遗留','降临','摇摆','击败','搏斗','降落','告诫','审查','害羞','脱离','应酬','打败','进攻','简易','平滑','高速','正规','详尽','威严','便宜','合格','好奇','清楚','全面','沉静','自如','茂密','坚决','威武','及时','合理','冷静','妥善','死刑','地狱','恶魔','叛徒','凶手','罪犯','魔鬼','罪恶','恶意','小偷','耻辱','灾害','火灾','奴才','棺材','诡计','疾病','悲剧','谣言','疯子','自杀','屠杀','暗杀','咒骂','侮辱','贪污','虐待','毁灭','欺骗','恐惧','背叛','怨恨','灭亡','恐吓','窒息','绑架','嫉妒','出卖','恶化','厌恶','下贱','卑劣','凶狠','可耻','窝囊','虚伪','自私','丑陋','无能','愚蠢','暴躁','刻薄','猖狂','残酷','狠心','悲哀','悲观','尖刻','羞耻','惊恐','张伟','王芳','李娜','张敏','王静','刘伟','张丽','李强','王敏','王磊','刘洋','王勇','李军','张勇','李杰','张磊','李娟','王军','张艳','张涛'];
      await this.setLine(1, '指导语')
      await this.setLine(2, '观看屏幕上的词语')
      await this.setLine(3, '如果词语是名字则需要按空格以继续');
      await this.setLine(4, '请尽量避免眨眼，吞咽口水等动作');
      await this.setLine(5, '如果理解指导语，按空格开始实验');
      await this.waitKey('Space');
      // await this.stop();
      await this.clearAll();
      this.ctx3 = 'font-size: 16rem';
      for(const i of seq.slice(0, 20)) {
        await this.setLine(3, '+');
        await this.logEvent(`${this.eegdata_pkg},word=${i},+`);
        await this.sleep(1200);
        await this.setLine(3, wordList[i-1]);
        await this.logEvent(`${this.eegdata_pkg},word=${i},start`);
        await this.sleep(200);
        if(this.id2class(i) == 4) {
          await this.waitKey('Space');
        } else {
          await this.sleep(1000);
          await this.logEvent(`${this.eegdata_pkg},word=${i},end`);
        }
      }
      // await this.setLine(2, '稍事休息');
      // await this.setLine(3, '下一个实验：请保持睁眼，并盯着中心的十字');
      // await this.setLine(4, '准备好按空格开始');
      // await this.waitKey('Space');
      // await this.clearAll();
      // await this.setLine(3, '+');
      // await this.logEvent(`${this.eegdata_pkg},eyesopen,start`);
      // await this.sleep(60*1000);
      // await this.logEvent(`${this.eegdata_pkg},eyesopen,end`);
      // await this.setLine(2, '稍事休息');
      // await this.setLine(3, '下一个实验：闭眼静息，听到滴的一声睁眼');
      // await this.setLine(4, '准备好按空格开始');
      // await this.waitKey('Space');
      // await this.clearAll();
      // await this.setLine(3, '请闭眼');
      // await this.logEvent(`${this.eegdata_pkg},eyesclose,start`);
      // await this.sleep(60*1000);
      // await this.logEvent(`${this.eegdata_pkg},eyesclose,end`);
      // await this.playMusic('di');
      this.ctx3 = '';
      await this.logEvent(`${this.eegdata_pkg},stop`);
      await this.stop();
    },
    async test_workflow() {
      await this.sleep(1000);
      await this.setLine(3, '请注视中心');
      await this.sleep(3000);
      await this.playVideo('Clip_1');
      await this.setLine(3, '+');
      await this.playMusic('di');
      await this.sleep(3000);
      await this.logEvent(`${this.eegdata_pkg},start`); // 正式开始采集
      await this.setLine(3, '好的，现在测试一下按空格');
      const t1 = await this.waitKey('Space');
      await this.setLine(3, `你按下了空格，用时${t1}毫秒`);
      const t2 = await this.waitKey('Space');
      await this.setLine(3, `你又按下了一次空格，用时${t2}毫秒`);
      const t3 = await this.waitKey('Space');
      await this.setLine(3, `你又又又按下了N次空格，用时${t3}毫秒`);
      await this.sleep(3000);
      await this.logEvent(`${this.eegdata_pkg},stop`); // 试验结束
      await this.setLine(3, '实验结束，好耶');
      await this.stop();
    },
    async ywt_workflow() {
      await this.setLine(3, '欢迎参加实验！');
      await this.sleep(3000);
      await this.setLine(3, '按空格开始实验');
      await this.waitKey('Space');
      await this.logEvent(`${this.eegdata_pkg},Start`);
      await this.setLine(3, '现在是静息实验，请根据提示睁眼或闭眼');
      await this.sleep(3000);
      await this.setLine(3, '请注视中心，并睁眼静息');
      await this.logEvent(`${this.eegdata_pkg},EyesOpen`);
      await this.sleep(3000);
      await this.setLine(3, '+');
      await this.sleep(2*60*1000); // 2min
      await this.setLine(3, '睁眼静息实验已经结束');
      await this.setLine(4, '下面进行闭眼静息实验，听到提示音后睁眼');
      await this.sleep(3000);
      await this.setLine(3, '现在开始闭眼静息');
      await this.setLine(4, '请闭眼');
      await this.logEvent(`${this.eegdata_pkg},EyesClose`);
      await this.sleep(2*60*1000); // 2min
      await this.playMusic('di');
      await this.setLine(3, '现在请睁眼，实验马上继续');
      await this.clearLine(4);
      await this.sleep(3000);
      await this.clearLine(3);
      await this.setLine(2, '下面进行运动想象实验');
      await this.setLine(3, '实验过程中保持身体不动，根据箭头提示进行对应想象');
      await this.setLine(4, '↑：舌头　　　↓：腿　　　←：左手　　　→：右手');
      await this.setLine(5, '调整坐姿后，按空格开始实验');
      await this.waitKey('Space');
      await this.clearLine(2);
      await this.clearLine(3);
      await this.clearLine(4);
      await this.clearLine(5);
      for(let i=0; i<2; ++i) {
        for(const { a, b, c } of [
          { a: '舌头', b: '↑', c: 1 },
          { a: '腿', b: '↓', c: 2 },
          { a: '左手', b: '←', c: 3 },
          { a: '右手', b: '→', c: 4 },
        ]) {
          await this.setLine(3, `请想象动${a}`);
          await this.logEvent(`${this.eegdata_pkg},Imagine=${c},Step=${i}`);
          await this.sleep(1000);
          await this.setLine(3, b);
          await this.sleep(90*1000); // 1min30s
          await this.logEvent(`${this.eegdata_pkg},ImagineEnd`);

          // 算式
          const p = ~~(Math.random()*100);
          const q = ~~(Math.random()*100);
          let r;
          if(Math.random() > 0.5) {
            r = p + q;
          } else {
            r = ~~(Math.random()*100);
          }
          await this.setLine(2, '请判断下面的等式是否成立');
          await this.setLine(3, `${p}+${q}=${r}`);
          this.logEvent(`${this.eegdata_pkg},${p}+${q}=${r},expect=${(p+q==r)?1:0}`);
          await this.setLine(4, '如果成立请按1，如果不成立请按0');
          await this.waitKey([ 'Numpad0', 'Numpad1' ]);

          await this.clearAll();
          await this.setLine(2, '好的，现在可以休息一会')
          await this.setLine(3, '调整舒服的姿势进行接下来的实验');
          await this.setLine(4, '如果准备好了，请按空格继续');
          await this.waitKey('Space');
          await this.clearAll();
        }
      }
      this.logEvent(`${this.eegdata_pkg},Stop`); // 试验结束
      await this.setLine(3, '实验结束，好耶');
      await this.stop();

    },
    async connect() {
      if(this.stage !== 0) return;
      this.stage = 1;
      console.log('连接设备');
      try {
        this.port = await navigator.serial.requestPort(); // https://wicg.github.io/serial/
        await this.port.open({ baudRate: 115200 });
        await this.sendString('AT+ROLE=0\r\n'); // 蓝牙初始化

        while(this.port.readable) {
          const reader = this.port.readable.getReader();

          try {
            while (true) {
              const { value, done } = await reader.read();
              console.log('value,done='+value, done);
              if (done) {
                reader.releaseLock();
                break;
              }
              if (value) {
                if(this.stage <= 5) {
                  for(const i of value) {
                    this.recvBuffer.push(i);
                    // meet \r\n
                    if(this.recvBuffer[this.recvBuffer.length - 2] === 13 && this.recvBuffer[this.recvBuffer.length - 1] === 10) {
                      console.log(this.recvBuffer);
                      const msg = this.recv2str(this.recvBuffer);
                      this.blueMsg += msg;
                      this.recvBuffer = [];
                      console.log(`(${this.stage}) >>>`, msg);
                      if(this.stage === 1 && msg === '+IM_MASTER\r\n') {
                        this.stage = 2;
                        this.blueMsg = '';
                      }
                      if(this.stage === 3 && msg == '+IM_READY\r\n') {
                        const regex = /YN_.+/g
                        const found = Array.from(new Set(this.blueMsg.match(regex)))
                        this.id = found[0]
                        console.log(`Found id=${this.id}`);
                        this.blueMsg = '';
                        this.stage = 4;
                      }
                      if(this.stage === 5 && msg === '+IM_CONNECT\r\n') {
                        this.blueMsg = '';
                        console.log('连接成功');
                        this.connected = true;
                      }
                    }
                  }
                }

                if(this.stage === 2) {
                  await this.sendString('AT+SCAN=1,5\r\n'); // 蓝牙扫描
                  this.stage = 3;
                }

                if(this.stage === 4) {
                  if(this?.id?.split('_')[1]) {
                    await this.sendString(`AT+CONN=0,${this.id.split('_')[1]}\r\n`); // 发送连接请求
                    //await this.sendString(`AT+CONN=0,01`); // 发送连接请求
                    this.stage = 5;
                  }
                }

                if(this.stage >= 6) {
                  for(const i of value) {
                    this.eegdata.push(i);
                    console.log(i);
                    console.log(this.eegdata);
                    if(++this.eegdata_step === 104) {
                      this.eegdata_step = 0;
                      const single_output = [];
                      for(let i=0; i<104; ++i) {
                        single_output.push(this.eegdata[this.eegdata_pkg*104+i]);
                      }
                      console.log(`#${this.eegdata_pkg}`, single_output);
                      this.eegdata_post = this.eegdata_post.concat(single_output);
                      ++this.eegdata_pkg;
                    }
                  }
                  if(this.eegdata_pkg && this.eegdata_pkg % 100 === 0) {
                    await this.saveEEG(this.post_pid, this.eegdata_post);
                    ++this.post_pid;
                    this.eegdata_post = [];
                  }
                }
              }
            }
          } catch (err) {
            console.error(err);
          }
        }

      } catch (err) {
        console.error(err);
      }
    },
    async fullscreen() {
      // shift to show
      this.guide = false;
      this.show = true;
      document.body.setAttribute('style', 'background: #000000; cursor: none;');

      if(screenfull.isEnabled && !screenfull.isFullscreen) {
        screenfull.toggle();
      } else {
        alert('浏览器不支持全屏，请按 F11 手动调整全屏');
      }
    },

    async data_up() {
      await this.sendHex([ 0x55, 0xAA, 0x01, 0x55, 0xAA ]); // 开始采集
    },
    async data_down() {
      await this.sendHex([ 0x55, 0xAA, 0x00, 0x55, 0xAA ]); // 暂停采集
      await this.sendHex([ 0xAA,  0xAA,  0x02,  0x0A,  0x64,  0x69,  0x73,  0x63,  0x6F,  0x6E,  0x6E,  0x65,  0x63,  0x74,  0xD5 ]); // 结束
      await this.sendHex([ 0xAA,  0xAA,  0x02,  0x0A,  0x64,  0x69,  0x73,  0x63,  0x6F,  0x6E,  0x6E,  0x65,  0x63,  0x74,  0xD5 ]);
    },
    async start() {
      this.stage = 6;
      await this.data_up();

      this.token = await this.getToken();
      console.log(`Token: ${this.token}`);

      // set fullscreen
      await this.fullscreen();
      console.log(`fullscreen`);
      // play execute
      await this.execute();
      console.log(`execute`);
    },
    async stop() {
      await this.data_down();
      await this.saveEEG(this.post_pid, this.eegdata_post, true); // send the last package
      // const u8array = new Uint8Array(this.eegdata);
      // const url = window.URL.createObjectURL(new Blob([ u8array ]));
      // const link = document.createElement('a');
      // link.style.display = 'none';
      // link.href = url;
      // link.setAttribute('download', `${this.token}.eeg`);
      // document.body.appendChild(link);
      // link.click();
      // document.body.removeChild(link);
      this.eegdata = [];
      this.eegdata_pkg = 0;
      this.eegdata_step = 0;
      // 跳转到结果页面
      console.log('goto result');
      location.href = `/result?token=${this.token}`;
    }
  },
  mounted() {
    document.body.setAttribute('style', 'background: #172E5A;');
  },
  components: {
    videoPlay
  }
}
</script>

<style scoped>
.button {
  margin: 0 10px;
}
.content{
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0px;
  left: 0px;
  bottom: 0px;
  right: 0px;
  margin: auto;
}
.word {
  position: relative;
  font-size: 5rem;
  text-align: center;
  color: #ffffff;
}
</style>