var express = require('express');
var router = express.Router();
var fs = require('fs');
const path = require('path');
var eeg = require('./eeg.js');
var fetch = require('node-fetch');

const tokensData = {}; // 存储每个token的请求时间戳数组

let textarea = '.';
let userSelected = 'temp';

const generateRandom = () => {
	return Math.random().toString(16).slice(2);  
}

console.log(__dirname);
const grandparentDir = path.resolve(__dirname, '../../../');
console.log(grandparentDir);
const savePath = path.join(grandparentDir, 'config.json')
try {
  const data = fs.readFileSync(savePath, 'utf8');
  console.log('读取文件成功:', data);
  textarea = data.replace(/"/g, '');
} catch (err) {
  console.error('读取文件失败:', err);
}
const savePath_ = path.join(grandparentDir, 'user.json')
try {
  const data = fs.readFileSync(savePath_, 'utf8');
  console.log('读取文件成功:', data);
  userSelected = data.replace(/"/g, '');
} catch (err) {
  console.error('读取文件失败:', err);
}

const saveurl = textarea+'/'+userSelected;
console.log(saveurl);
var open = true;

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

const buf2hex = (buffer) => {
  return Array.prototype.map.call(new Uint8Array(buffer), x => ('00' + x.toString(16)).slice(-2)).join('');
}

const hex2buf = (hex) => {
  return new Uint8Array(hex.match(/[\da-f]{2}/gi).map(function (h) {
    return parseInt(h, 16)
  }))?.buffer;
}

router.get('/getToken', (req, res) => {
  const token = generateRandom();
  console.log(token);
  try {
    if(open) {
      fs.mkdirSync(`${saveurl}/data/${token}`, { recursive: true });
      fs.mkdirSync(`${saveurl}/data/${token}/eeg`, { recursive: true });
      fs.writeFileSync(`${saveurl}/data/${token}/createTime.txt`, (new Date().getTime().toString()));
      console.log(saveurl+'/data/'+token);
    }}catch (err) {
      console.error('File write failed:', err);
    }
  
  res.send(JSON.stringify({ code: 200, token }));
});

router.post('/event', (req, res) => {
  const token = req.query?.token;
  //console.log(req.body);
  if(open) {
    // fs.appendFileSync(`${saveurl}/data/${token}/log.txt`, ((new Date().getTime().toString()) + ',' + req.body?.event + '\n'));
    fs.appendFileSync(`${saveurl}/data/${token}/log.txt`, (new Date().toISOString() + ',' + req.body?.event + '\n'));
  }
  res.send(JSON.stringify({ code: 200 }));
});

router.post('/saveEEG', (req, res) => {
  const token = req.query?.token;
  const pid = req.query?.pid;
  const end = req.query?.end;
  console.log(pid);
  if(req.body.data) {
    const data = hex2buf(req.body?.data);
    console.log(data);
    if(open) {
      fs.writeFileSync(`${saveurl}/data/${token}/eeg/${pid}.eeg`, new DataView(data));
      //fs.appendFileSync(`${saveurl}/data/${token}/log.txt`, (new Date().toISOString() + ',' + '[savedata]' + '\n'));
    }
  }
  if(open && end) { // 合并 eeg 文件
    for(let i = 0; i <= pid; ++i) {
      if(fs.existsSync(`${saveurl}/data/${token}/eeg/${i}.eeg`)) {
        const r = fs.readFileSync(`${saveurl}/data/${token}/eeg/${i}.eeg`)
        fs.appendFileSync(`${saveurl}/data/${token}/data.eeg`, r);
      }
    }
    // 然后转码成json保存下来
    
  }
  res.send(JSON.stringify({ code: 200 }));
});

// 计算大五人格，调用python，传入id，得到ocean
router.get('/calc', (req, res) => {

  const token = req.query?.token;

  // const eegData = fs.readFileSync(`${saveurl}/data/${token}/data.eeg`);
  // const createTime = fs.readFileSync(`${saveurl}/data/${token}/createTime.txt`, 'utf-8');

  // // 解析分段
  // const logtxt = fs.readFileSync(`${saveurl}/data/${token}/log.txt`, 'utf-8').split('\n');
  // const wdinfo = {};
  // for(let i in logtxt) {
  //   const subarr = logtxt[i].split(',');
  //   if(subarr.length >= 4) {
  //     const wordID = subarr[2].split('=')[1];
  //     if(subarr[2] == 'resolveKey=Space') {
  //       const i_1 = logtxt[i-1].split(',');
  //       const wID = i_1[2]?.split('=')[1];
  //       if(wID && ~~wID == wID) {
  //         if(!wdinfo[wID]) {
  //           wdinfo[wID] = {
  //             startTime: 0,
  //             endTime: subarr[0]
  //           };
  //         } else {
  //           wdinfo[wID].endTime = subarr[0];
  //         }
  //       }
  //     }
  //     if(!wordID || ~~wordID != wordID) continue;
  //     if(subarr[3] == 'start') {
  //       if(!wdinfo[wordID]) {
  //         wdinfo[wordID] = {
  //           startTime: subarr[0],
  //           endTime: 0
  //         };
  //       } else {
  //         wdinfo[wordID].startTime = subarr[0];
  //       }
  //     } else if(subarr[3] == 'end') {
  //       if(!wdinfo[wordID]) {
  //         wdinfo[wordID] = {
  //           startTime: 0,
  //           endTime: subarr[0]
  //         };
  //       } else {
  //         wdinfo[wordID].endTime = subarr[0];
  //       }
  //     }
  //   }
  // }
  // // console.log(wdinfo);

  // const eegStream = eeg.batchData(eegData);
  // // console.log(eegStream.data);
  // let data_ch1 = eegStream.data?.ch1;
  // let data_ch2 = eegStream.data?.ch2;

  // const data = {};

  // for(let i in wdinfo) {
  //   const startBound = ~~ ((wdinfo[i].startTime - createTime) / 2);
  //   const endBound = ~~ ((wdinfo[i].endTime - createTime) / 2);
  //   data[i] = {
  //     ch1: data_ch1.slice(startBound, endBound),
  //     ch2: data_ch2.slice(startBound, endBound)
  //   };
  // }

  // // // 切段，最终保存的是若干个片段的json
  // fs.writeFileSync(`${saveurl}/data/${token}/data.json`, JSON.stringify(data));


  // fetch python
  fetch(`http://127.0.0.1:8080?token=${token}`).then((res) => {
    if(res.status === 200) {
      return res.json();
    } else {
      return Promise.reject(res.json());
    }
  }).then((data) => {
    res.send(JSON.stringify({ code: 200, result: data }));
  }).catch((err) => {
    console.error(err);
  });
});


module.exports = router;
