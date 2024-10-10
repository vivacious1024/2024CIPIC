<template>
	<br>
	<img class="hero-logo" src="../assets/icon.png" /><br>
	<h2 class="hero-text">STD::EEG</h2>
	<p class="hero-tagline">一个标准的脑机接口平台，实现了对设备和任务的解耦</p>

	<div class="links">
		<div class="link-item">
			<a href="/aubot">使用文档</a>
		</div>
		<div class="link-item link-dot">•</div>
		<div class="link-item">
			<a href="/question">常见问题</a>
		</div>
		<div class="link-item link-dot">•</div>
		<div class="link-item">
			<a href="/team"> 开发团队 </a>
		</div>
	</div>

	<div class="features">
		<div class="feature-item">
			<article>
				<h2 class="title">自定义脑电任务</h2>
				<p class="detail">可自定义脑电任务，在采集过程中为脑电自动打标签</p>
			</article>
		</div>
		<div class="feature-item">
			<article>
				<h2 class="title">可任意适配脑电设备</h2>
				<p class="detail">接口标准，可通过插件的方式实现接入不同的脑电设备</p>
			</article>
		</div>
		<div class="feature-item">
			<article>
				<h2 class="title">实时采集分析脑电数据</h2>
				<p class="detail">可实时采集并分析脑电数据，内置多种脑电分析算法</p>
			</article>
		</div>
		<div class="feature-item">
			<article>
				<h2 class="title">跨平台支持</h2>
				<p class="detail">底层采用 Electron，可在 Windows、MacOS、Linux 上运行</p>
			</article>
		</div>
		<div class="feature-item">
			<article>
				<h2 class="title">测评工作流</h2>
				<p class="detail">支持创建测评工作流，为您的脑电测评带来更好地体验</p>
			</article>
		</div>
		<div class="feature-item">
			<article>
				<h2 class="title">支持多种结果反馈</h2>
				<p class="detail">支持多种图表展示，可导出测评的结果数据</p>
			</article>
		</div>
	</div>
</template>

<script>
export default {
	name: 'About'
}
</script>

<style lang="less" scoped>
* {
	padding: 0;
	margin: 0;
}

.hero-logo {
	margin: auto;
	width: 25%;
	height: 25%;
	transition: all 0.3s;
}

@media (max-width: 840px) {
	.versions {
		display: none;
	}

	.hero-logo {
		margin-top: -1.5rem;
	}
}

.hero-text {
	font-weight: 400;
	color: #3bb18e;
	text-align: center;
	margin-top: -0.5rem;
	margin-bottom: 10px;
}

@media (max-width: 660px) {
	.hero-logo {
		display: none;
	}

	.hero-text {
		margin-top: 20px;
	}
}

.hero-tagline {
	text-align: center;
	margin-bottom: 14px;
}

.links {
	display: flex;
	align-items: center;
	justify-content: center;
	margin-bottom: 24px;
	font-size: 18px;
	font-weight: 500;

	a {
		font-weight: 500;
	}

	.link-item {
		padding: 0 4px;
	}
}

.features {
	display: flex;
	flex-wrap: wrap;

	.feature-item {
		width: 33.33%;
		box-sizing: border-box;
	}

	article {
		background-color: rgba(194, 245, 255, 0.1);
		border-radius: 8px;
		box-sizing: border-box;
		padding: 12px;
		height: 100%;
	}

	span {
		color: #d4e8ef;
		word-break: break-all;
	}

	.title {
		font-size: 17px;
		font-weight: 500;
		color: #3bb18e;
		line-height: 22px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.detail {
		font-size: 14px;
		font-weight: 500;
		line-height: 22px;
		margin-top: 6px;
	}
}

@media (max-width: 660px) {
	.features .feature-item {
		width: 50%;
	}
}

@media (max-width: 480px) {
	.links {
		flex-direction: column;
		line-height: 32px;

		.link-dot {
			display: none;
		}
	}

	.features .feature-item {
		width: 100%;
	}
}
</style>