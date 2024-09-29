<template>
  <div id="guide" style="margin: 0 20rem 0 20rem;">
    <div style="text-align: center;">
      <!-- <n-gradient-text type="info"> -->
        <h1 style="color: #ffffff; font-size: 2rem;">测评结果</h1>
      <!-- </n-gradient-text> -->
    </div>

    <table style="border: 0; margin: 0 auto;">
      <tr>
        <th><div id="graph" style="width: 600px; height:400px;"></div></th>
        <th><img :src="maxsrc" style="height: 400px"></th>
      </tr>
  </table>
    <!-- <n-timeline horizontal style="margin-top: 1rem; justify-content: center;">
      <n-timeline-item
        type="success"
        title="采集数据"
        content="佩戴设备并连接"
      />
      <n-timeline-item
        :type="result.result?'success':'info'"
        title="分析数据"
        content="服务器进行分析"
      />
      <n-timeline-item
        :type="result.result?'success':''"
        title="得出结果"
        content="查看分析报告"
      />
    </n-timeline> -->

    <div style="text-align: center; margin-top: 1rem; color: #ffffff; font-size: 2rem;" v-if="!result.result">
      结果计算中……
    </div>
    <!-- <n-card title="测评结果" id="res" v-if="result.result">
      <n-row>
        <n-col :span="12">
          <n-statistic label="开放性">
            {{ result?.result?.score_o }}
          </n-statistic>
        </n-col>
        <n-col :span="12">
          <n-statistic label="尽责性">
            {{ result?.result?.score_c }}
          </n-statistic>
        </n-col>
        <n-col :span="12">
          <n-statistic label="外向型">
            {{ result?.result?.score_e }}
          </n-statistic>
        </n-col>
        <n-col :span="12">
          <n-statistic label="宜人性">
            {{ result?.result?.score_a }}
          </n-statistic>
        </n-col>
        <n-col :span="12">
          <n-statistic label="神经质">
            {{ result?.result?.score_n }}
          </n-statistic>
        </n-col>
        <n-col :span="12">
          <n-statistic label="专注度">
            {{ result?.result?.score_con }}
          </n-statistic>
        </n-col>
      </n-row>
    </n-card><br> -->
    <n-card v-if="result.result">
      <span style="font-size: 2em; white-space: pre-wrap;">
        {{ comment }}
      </span>
    </n-card>
  </div>
</template>


<script>
import * as echarts from 'echarts';

export default {
  data() {
    let token = this.$route.query.token;;
    let result = {};
    let max = 0;
    let maxsrc = '';
    let comment = '';
    return { token, result, max, maxsrc, comment };
  },
  methods: {
    async getResult(token) {
      const res = await fetch(`http://localhost:3000/calc?token=${token}`);
      console.log('getresult');
      const result = await res.json();
      return result;
    },
  },
  async mounted() {
    document.body.setAttribute('style', 'background: #172E5A;');
    console.log(this.token);
    this.result = await this.getResult(this.token);
    console.log(this.result);

    if(this.result?.result?.score_o > this.max) this.max = this.result?.result?.score_o;
    if(this.result?.result?.score_c > this.max) this.max = this.result?.result?.score_c;
    if(this.result?.result?.score_e > this.max) this.max = this.result?.result?.score_e;
    if(this.result?.result?.score_a > this.max) this.max = this.result?.result?.score_a;
    if(this.result?.result?.score_n > this.max) this.max = this.result?.result?.score_n;

    if(Math.abs(this.max - this.result?.result?.score_o) < 0.000001) this.max = 'o';
    if(Math.abs(this.max - this.result?.result?.score_c) < 0.000001) this.max = 'c';
    if(Math.abs(this.max - this.result?.result?.score_e) < 0.000001) this.max = 'e';
    if(Math.abs(this.max - this.result?.result?.score_a) < 0.000001) this.max = 'a';
    if(Math.abs(this.max - this.result?.result?.score_n) < 0.000001) this.max = 'n';

    this.maxsrc = this.max + '.jpg';

    if(this.max == 'o') {
      this.comment = 
`    您的开放性评分最高。开放性描述的是一个人的认知风格和探索新事物的态度，其中包括了想象力、审美、感受性、尝新、求知欲、价值观六个方面。
    开放性高的人，愿意接受新鲜的观念和事物；开放性低的人更务实、从众、不喜欢复杂和抽象的事物。
    一般像作家、科学家、设计师等职业，开放性得分会比较高；开放性低分的人适宜从事警察、会计、护士等需要严格遵循既有规范的职业。
`;
    } else if(this.max == 'c') {
      this.comment = 
`    您的尽责性评分最高。尽责性指的是我们控制、管理和调节自身冲动的方式，它可以分为六个子维度，自我效能、条理性、责任感、成就愿望、意志力和审慎。
    尽责性高的人，勤奋、自律、有条理、道德要求高、值得信赖，看起来有些过于警惕；尽责性低分的人，懒惰、无组织性、易冲动、行为缺乏一致性，给人不值得信赖的观感。
    尽责性高的人适合负责一些大项目，如科学研究、计算机、精密器械制造等。
`;
    } else if(this.max == 'e') {
      this.comment = 
`    您的外向性评分最高。外向性是指人际活动的数量和密度以及对刺激的需要和获得愉悦的能力，其中包含热情、乐群性、独断性、活力、寻求刺激、积极情绪这六个方面。
    外向性高的人，喜欢与人接触并充满活力，他们热爱冒险、挑战、追求刺激，喜欢受到他人关注；外向性低的人，往往安静、谨慎，但并不等同于害羞、腼腆，他们只是更爱独处，不需要从外界感受过多的刺激。
    外向性高分的人适合选择一些和人打交道的职业，如人力资源和销售主管或主持类职位等；外向性低分的人适合选择一些能独立完成工作的职业，如作家、画家、篆刻、科研工作者等。
`;
    } else if(this.max == 'a') {
      this.comment = 
`    您的宜人性评分最高。宜人性考察的是个体对他人所持的态度。它代表了“爱”，以及对合作和人际和谐是否看重，其中包含信任、坦诚、利他、顺从、谦逊和同理心六个维度。
    宜人性高分的人，对人真诚、友善、宽容，他们富有同情心，认为世界充满爱；宜人性低分的人，谨慎、猜忌、防备心理较重，他们往往将他人放在敌对立场，愤世嫉俗。
    宜人性高分的人适合从事一些为他人服务的职业，如秘书、助理、心理辅导员等；宜人性低分的人非常理性，适合科学、工程、军事等此类要求客观决策的情境，如科学家、评论家、士兵等。
`;
    } else if(this.max == 'n') {
      this.comment = 
`    您的神经质评分最高。神经质是指一个人在调节情绪的过程中，是否倾向于消极或是不稳定情绪，其中包括了焦虑、愤怒与敌意、忧郁、自我意识、冲动性、脆弱性。
    神经质高分的人，往往体验着较大的压力，他们的情绪稳定性差，更易体会到负面情绪；神经质低分的人，往往情绪稳定，烦恼较少，应对压力的能力较强。
    神经质高分的人要学会调节自己的情绪，适合从事需要迅速判断做决定的工作。
`;
    } 

    if(this.result?.result?.score_con > 0.6) {
      this.comment += `    您在实验的过程中专注度不错，为${this.result?.result?.score_con}（满分为1），这是一件好事，表明您此次的结果应更符合真实情况。`;
    } else if(this.result?.result?.score_con > 0.3) {
      this.comment += `    您在实验的过程中专注度一般，为${this.result?.result?.score_con}（满分为1）。`;
    } else {
      this.comment += `    您在实验的过程中专注度不好，为${this.result?.result?.score_con}（满分为1），这表明您此次的结果将与真实情况有更多出入。`;
    }

    setTimeout(() => {
      const chartDom = document.getElementById('graph');
      const myChart = echarts.init(chartDom);

      myChart.setOption({
        xAxis: {
          type: 'category',
          data: ['开放性', '尽责性', '外向性', '宜人性', '神经质'],
          axisLine: {
            lineStyle: {
              color: '#ffffff',
              width: 0,
              type: 'solid'
            }
          }
        },
        yAxis: {
          type: 'value',
          axisLine: {
            lineStyle: {
              color: '#ffffff',
              width: 0,
              type: 'solid'
            }
          }
        },
        series: [
          {
            data: [
              {
                value: this.result?.result?.score_o,
                itemStyle: {
                  color: '#F5CB27'
                }
              },
              {
                value: this.result?.result?.score_c,
                itemStyle: {
                  color: '#BB9BCA'
                }
              },
              {
                value: this.result?.result?.score_e,
                itemStyle: {
                  color: '#D74F67'
                }
              },
              {
                value: this.result?.result?.score_a,
                itemStyle: {
                  color: '#19935A'
                }
              },
              {
                value: this.result?.result?.score_n,
                itemStyle: {
                  color: '#A1E3EF'
                }
              },
            ],
            type: 'bar',
            markPoint : {
              data : [
                  { value : this.result?.result?.score_o.toFixed(3), xAxis: '开放性', yAxis: this.result?.result?.score_o },
                  { value : this.result?.result?.score_c.toFixed(3), xAxis: '尽责性', yAxis: this.result?.result?.score_c },
                  { value : this.result?.result?.score_e.toFixed(3), xAxis: '外向性', yAxis: this.result?.result?.score_e },
                  { value : this.result?.result?.score_a.toFixed(3), xAxis: '宜人性', yAxis: this.result?.result?.score_a },
                  { value : this.result?.result?.score_n.toFixed(3), xAxis: '神经质', yAxis: this.result?.result?.score_n },
              ]
            }
          }
        ]
      });
    }, 100);
  },
  async created() {
    
  },
  components: {
    
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
  font-size: 3rem;
  text-align: center;
  color: #ffffff;
}
</style>