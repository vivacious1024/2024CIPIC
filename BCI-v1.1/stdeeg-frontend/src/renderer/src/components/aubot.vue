<template>
	<div class="container">
		<button class="back-button" @click="goBack">返回上一级</button>
		<div class="team-section">
			关于
		</div>
	</div>
</template>

<script setup>
const goBack = () => {
	window.history.back(); // 返回上一页
}
</script>

<style lang="less" scoped>
.container {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	padding: 20px;
}

.back-button {
	background-color: #00bcd4;
	color: white;
	padding: 10px 20px;
	border: none;
	border-radius: 5px;
	cursor: pointer;
	font-size: 16px;
	margin-bottom: 20px;
	transition: background-color 0.3s;

	&:hover {
		background-color: #0097a7;
		/* 鼠标悬停时改变背景色 */
	}
}

.team-section {
	font-size: 24px;
	color: #333;
}
</style>
