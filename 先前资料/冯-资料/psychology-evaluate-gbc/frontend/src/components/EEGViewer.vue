<template>
  <div id="main"></div>
</template>

<script>
import * as echarts from 'echarts';

export default {
  mounted() {
    // Object.defineProperty(document.getElementById('main'), 'clientWidth', { get: () => this?.width || 640 });
    // Object.defineProperty(document.getElementById('main'), 'clientHeight', { get: () => this?.height || 480 });

    const eegViewer = echarts.init(document.getElementById('main'));
    eegViewer.setData = ({ ch1, ch2 }) => {
      const axis = [];
      for(const i in ch1) {
        axis.push(i);
      }
      eegViewer.setOption({
        xAxis: {
          data: axis
        },
        yAxis: {},
        series: [
          {
            data: ch1,
            type: 'line',
            emphasis: {
              label: {
                show: true
              }
            },
            label: {
              show: false,
              position: 'top',
              fontSize: 20
            }
          },
          {
            data: ch2,
            type: 'line',
            emphasis: {
              label: {
                show: true
              }
            },
            label: {
              show: false,
              position: 'top',
              fontSize: 20
            }
          }
        ],
        toolbox: {
          show: true,
          feature: {
            dataZoom: {
              yAxisIndex: 'none'
            }
          }
        },
        dataZoom: [
        // 1.横向使用滚动条
        {
          type: 'slider',// 有单独的滑动条，用户在滑动条上进行缩放或漫游。inside是直接可以是在内部拖动显示
          show: true,// 是否显示 组件。如果设置为 false，不会显示，但是数据过滤的功能还存在。
          start: 0, // 数据窗口范围的起始百分比0-100
          end: 100, // 数据窗口范围的结束百分比0-100
          xAxisIndex: [0], // 此处表示控制第一个xAxis，设置 dataZoom-slider 组件控制的 x轴 可是已数组[0,2]表示控制第一，三个；xAxisIndex: 2 ，表示控制第二个。yAxisIndex属性同理
        },
        // 2.在内部可以横向拖动
        {
          type: 'inside', // 内置于坐标系中
          start: 0,
          end: 100,
          xAxisIndex: [0]
        },
        // // 3.纵向使用滚动条
        // {
        //   type: 'slider',
        //   show: true,
        //   yAxisIndex: [0], // 设置组件控制的y轴
        //   left: '93%', // 距离左侧的距离 可以使百分比，也可以是像素 left: '30'（30像素）
        //   start: 0,
        //   end: 100
        // },
        // // 4.在内部可以纵向拖动
        // {
        //   type: 'inside',
        //   yAxisIndex: [0],
        //   start: 0,
        //   end: 100
        // }
      ],
      });
    }
    window.eegViewer = eegViewer;
    window.onresize = () => {
      eegViewer.resize();
    };
  }
}
</script>

<style scoped>
  #main,
  html,
  body {
    width: 100%;
  }
  #main {
    height: 800px;
  }
</style>