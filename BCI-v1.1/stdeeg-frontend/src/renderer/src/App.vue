<template>
  <div id="app">
    <router-view />
    <!-- <HomeView/> -->
  </div>
</template>

<script>
//import HomeView from './views/HomeView.vue';
export default {
  name: 'App',
  components: { // 在这里注册想要在本页面中使用的子组件
    //HomeView
  },
  created() {
    window.addEventListener('keyup', this.keyUp, false)
  },
  methods: {
    keyUp(e) {
      if (e.code === 'F1') {
        if (this.$router.currentRoute.value.path === '/') {
          this.$router.push('/about');
        } else if (this.$router.currentRoute.value.path === '/about') {
          this.$router.push('/');
        }
      }
    },
  }
}
</script>
<style>
/*code {
  font-weight: 600;
  padding: 3px 5px;
  border-radius: 2px;
  background-color: #26282e;
  font-family:
    ui-monospace,
    SFMono-Regular,
    SF Mono,
    Menlo,
    Consolas,
    Liberation Mono,
    monospace;
  font-size: 85%;
}
a {
  color: #9feaf9;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  outline: none;
}
a:hover {
  border-bottom: 1px solid;
}*/
body {
  display: flex;
  flex-direction: column;
  font-family:
    Roboto,
    -apple-system,
    BlinkMacSystemFont,
    'Helvetica Neue',
    'Segoe UI',
    'Oxygen',
    'Ubuntu',
    'Cantarell',
    'Open Sans',
    sans-serif;
  color: #86a5b1;
  background-color: #e3f3f6;
  margin-left: 0;
  width: 100vw;
}
#app {
  /*flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 840px;
  margin: 0 auto;
  padding: 15px 30px 0 30px;*/
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>