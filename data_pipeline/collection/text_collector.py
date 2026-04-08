"""
文本数据采集器
支持从Common Crawl、Wikipedia、arXiv、GitHub等源采集文本数据
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, List, Dict, Any, Optional
import asyncio
import aiohttp
import logging
from pathlib import Path
import gzip
import re
import uuid
from datetime import datetime

from .base_collector import (
    BaseCollector, CollectionConfig, DataSource, 
    RawDataItem, DataType, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class TextCollectionConfig(CollectionConfig):
    """文本采集专用配置"""
    min_text_length: int = 50
    max_text_length: int = 100000
    languages: List[str] = field(default_factory=lambda: ['en', 'zh'])
    enable_language_detection: bool = True
    remove_boilerplate: bool = True
    filter_duplicate_lines: bool = True
    max_duplicate_line_ratio: float = 0.3
    # Common Crawl配置
    warc_batch_size: int = 100
    # Wikipedia配置
    wikipedia_dump_url: str = "https://dumps.wikimedia.org"
    # arXiv配置
    arxiv_batch_size: int = 100
    # GitHub配置
    github_token: Optional[str] = None
    min_stars: int = 10


class TextCollector(BaseCollector):
    """
    大规模文本语料采集器。
    数据来源：Common Crawl、Wikipedia、arXiv、GitHub代码、
             书籍语料(Project Gutenberg)、新闻语料。
    """
    
    def __init__(
        self,
        config: TextCollectionConfig,
        storage_client=None,
        metadata_store=None
    ):
        super().__init__(config, storage_client, metadata_store)
        self.text_config = config
        self.session = None
        
    async def _init_session(self):
        """初始化HTTP会话"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """关闭资源"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def collect(self, source: DataSource) -> AsyncIterator[RawDataItem]:
        """
        从指定数据源采集文本数据
        
        Args:
            source: 数据源配置
            
        Yields:
            RawDataItem: 文本数据项
        """
        await self._init_session()
        
        if source.source_type == DataSourceType.COMMON_CRAWL:
            async for item in self.collect_common_crawl(
                source.source_url or [],
                source.filters
            ):
                yield item
        
        elif source.source_type == DataSourceType.WIKIPEDIA:
            async for item in self.collect_wikipedia(
                source.metadata.get('lang', 'en'),
                source.metadata.get('dump_date', 'latest')
            ):
                yield item
        
        elif source.source_type == DataSourceType.ARXIV:
            async for item in self.collect_arxiv(
                source.metadata.get('categories', [])
            ):
                yield item
        
        elif source.source_type == DataSourceType.GITHUB:
            async for item in self.collect_github_code(
                source.metadata.get('languages', []),
                source.metadata.get('min_stars', self.text_config.min_stars)
            ):
                yield item
        
        else:
            self.logger.warning(f"Unsupported source type: {source.source_type}")
    
    async def collect_common_crawl(
        self,
        warc_paths: List[str],
        filters: Dict[str, Any] = None
    ) -> AsyncIterator[RawDataItem]:
        """
        从Common Crawl WARC文件中提取文本
        使用warcio库解析WARC格式
        
        Args:
            warc_paths: WARC文件路径列表
            filters: 过滤条件
            
        Yields:
            RawDataItem: 文本数据项
        """
        from warcio import ArchiveIterator
        
        for warc_path in warc_paths:
            try:
                self.logger.info(f"Processing WARC file: {warc_path}")
                
                # 下载WARC文件
                if warc_path.startswith('http'):
                    async with self.session.get(warc_path) as response:
                        if response.status == 200:
                            warc_content = await response.read()
                        else:
                            self.logger.error(f"Failed to download WARC: {response.status}")
                            continue
                else:
                    # 本地文件
                    with open(warc_path, 'rb') as f:
                        warc_content = f.read()
                
                # 解析WARC文件
                count = 0
                for record in ArchiveIterator(warc_content):
                    if record.rec_type == 'response':
                        # 提取HTML内容
                        html_content = record.content_stream().read()
                        
                        # 从HTML提取文本
                        text = self.extract_text_from_html(html_content.decode('utf-8', errors='ignore'))
                        
                        if text and len(text) >= self.text_config.min_text_length:
                            # 语言检测
                            if self.text_config.enable_language_detection:
                                lang = self.detect_language(text)
                                if lang not in self.text_config.languages:
                                    continue
                            
                            item = RawDataItem(
                                data_id=str(uuid.uuid4()),
                                data_type=DataType.TEXT,
                                content=text,
                                metadata={
                                    'url': record.rec_headers.get_header('WARC-Target-URI', ''),
                                    'length': len(text),
                                    'language': lang if self.text_config.enable_language_detection else 'unknown',
                                    'source': 'common_crawl'
                                },
                                source=DataSource(
                                    source_type=DataSourceType.COMMON_CRAWL,
                                    source_url=warc_path
                                )
                            )
                            
                            if self.validate_item(item):
                                count += 1
                                yield item
                                
                                if count % self.config.batch_size == 0:
                                    self.report_progress(count, -1, self.error_count)
                
                self.logger.info(f"Extracted {count} text items from {warc_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing WARC file {warc_path}: {e}")
                self.error_count += 1
    
    async def collect_wikipedia(
        self,
        lang: str = 'en',
        dump_date: str = 'latest'
    ) -> AsyncIterator[RawDataItem]:
        """
        下载并解析Wikipedia dump
        使用mwparserfromhell提取纯文本
        
        Args:
            lang: 语言代码
            dump_date: dump日期
            
        Yields:
            RawDataItem: 文本数据项
        """
        import mwparserfromhell
        import bz2
        
        # 构建Wikipedia dump URL
        dump_url = f"{self.text_config.wikipedia_dump_url}/{lang}wiki/{dump_date}/"
        
        try:
            # 查找最新的dump文件
            async with self.session.get(dump_url) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to access Wikipedia dump: {response.status}")
                    return
                
                html = await response.text()
                # 解析页面找到dump文件
                # 这里简化处理，实际需要解析HTML找到正确的文件
                
            # 下载并解析dump文件 (示例使用pages-articles.xml.bz2)
            dump_file = f"{lang}wiki-{dump_date}-pages-articles.xml.bz2"
            dump_url = f"{dump_url}{dump_file}"
            
            self.logger.info(f"Downloading Wikipedia dump: {dump_url}")
            
            async with self.session.get(dump_url) as response:
                if response.status == 200:
                    # 流式解压和解析
                    decompressor = bz2.BZ2Decompressor()
                    buffer = b''
                    
                    async for chunk in response.content.iter_chunked(8192):
                        try:
                            buffer += decompressor.decompress(chunk)
                            
                            # 解析XML并提取文本
                            # 这里需要XML解析器和mwparserfromhell处理
                            # 简化示例，实际实现需要完整解析流程
                            
                        except Exception as e:
                            self.logger.error(f"Error decompressing: {e}")
                            continue
                
                else:
                    self.logger.error(f"Failed to download Wikipedia dump: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting Wikipedia: {e}")
            self.error_count += 1
    
    async def collect_arxiv(
        self,
        categories: List[str] = None
    ) -> AsyncIterator[RawDataItem]:
        """
        通过arXiv Bulk Data Access获取论文全文
        使用MinerU或PyPDF2提取PDF文本
        
        Args:
            categories: 论文类别列表，如['cs.AI', 'cs.LG']
            
        Yields:
            RawDataItem: 文本数据项
        """
        import feedparser
        
        # arXiv API endpoint
        arxiv_api_url = "http://export.arxiv.org/api/query?"
        
        # 构建查询
        if categories:
            query = " OR ".join([f"cat:{cat}" for cat in categories])
        else:
            query = "all"
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': self.text_config.arxiv_batch_size
        }
        
        try:
            # 查询arXiv
            url = arxiv_api_url + "&".join([f"{k}={v}" for k, v in params.items()])
            async with self.session.get(url) as response:
                if response.status == 200:
                    feed_content = await response.text()
                    feed = feedparser.parse(feed_content)
                    
                    for entry in feed.entries:
                        try:
                            # 提取论文信息
                            title = entry.title
                            abstract = entry.summary
                            pdf_url = entry.link.replace('abs', 'pdf')
                            
                            # 下载PDF并提取文本
                            # 这里简化处理，仅返回标题和摘要
                            text = f"{title}\n\n{abstract}"
                            
                            if len(text) >= self.text_config.min_text_length:
                                item = RawDataItem(
                                    data_id=str(uuid.uuid4()),
                                    data_type=DataType.TEXT,
                                    content=text,
                                    metadata={
                                        'title': title,
                                        'arxiv_id': entry.id.split('/')[-1],
                                        'categories': [tag.term for tag in entry.tags],
                                        'published': entry.published,
                                        'source': 'arxiv'
                                    },
                                    source=DataSource(
                                        source_type=DataSourceType.ARXIV,
                                        source_url=entry.id
                                    )
                                )
                                
                                if self.validate_item(item):
                                    yield item
                                    
                        except Exception as e:
                            self.logger.error(f"Error processing arXiv entry: {e}")
                            continue
                else:
                    self.logger.error(f"Failed to query arXiv: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting arXiv: {e}")
            self.error_count += 1
    
    async def collect_github_code(
        self,
        languages: List[str] = None,
        min_stars: int = 10
    ) -> AsyncIterator[RawDataItem]:
        """
        通过GitHub API获取开源代码
        需要GitHub Token以避免rate limit
        
        Args:
            languages: 编程语言列表
            min_stars: 最小star数
            
        Yields:
            RawDataItem: 代码数据项
        """
        # GitHub搜索API
        github_api_url = "https://api.github.com/search/repositories"
        
        headers = {}
        if self.text_config.github_token:
            headers['Authorization'] = f"token {self.text_config.github_token}"
        
        # 构建查询
        language_query = " OR ".join([f"language:{lang}" for lang in languages]) if languages else ""
        query = f"stars:>={min_stars} {language_query}".strip()
        
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 100
        }
        
        try:
            async with self.session.get(
                github_api_url,
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for repo in data.get('items', []):
                        try:
                            # 获取仓库内容
                            contents_url = repo['contents_url'].replace('{+path}', '')
                            
                            # 递归获取代码文件
                            # 这里简化处理，仅返回仓库信息
                            code_text = f"""
Repository: {repo['full_name']}
Description: {repo.get('description', '')}
Language: {repo.get('language', '')}
Stars: {repo['stargazers_count']}
URL: {repo['html_url']}
"""
                            
                            item = RawDataItem(
                                data_id=str(uuid.uuid4()),
                                data_type=DataType.TEXT,
                                content=code_text,
                                metadata={
                                    'repo_name': repo['full_name'],
                                    'language': repo.get('language'),
                                    'stars': repo['stargazers_count'],
                                    'forks': repo['forks_count'],
                                    'source': 'github'
                                },
                                source=DataSource(
                                    source_type=DataSourceType.GITHUB,
                                    source_url=repo['html_url']
                                )
                            )
                            
                            if self.validate_item(item):
                                yield item
                                
                        except Exception as e:
                            self.logger.error(f"Error processing GitHub repo: {e}")
                            continue
                else:
                    self.logger.error(f"Failed to query GitHub: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error collecting GitHub code: {e}")
            self.error_count += 1
    
    async def collect_books(self) -> AsyncIterator[RawDataItem]:
        """
        从Project Gutenberg获取公开书籍文本
        
        Yields:
            RawDataItem: 书籍文本数据项
        """
        # Project Gutenberg镜像站点
        gutenberg_url = "https://www.gutenberg.org/"
        
        # 实现书籍采集逻辑
        # 这里简化处理，实际需要解析Gutenberg目录并下载
        pass
    
    def extract_text_from_html(self, html: str) -> str:
        """
        使用trafilatura库从HTML中提取正文文本
        
        Args:
            html: HTML内容
            
        Returns:
            str: 提取的纯文本
        """
        try:
            import trafilatura
            
            # 使用trafilatura提取正文
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            
            if text:
                # 清理文本
                text = self._clean_text(text)
                return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML: {e}")
        
        return ""
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 移除连续重复的行
        if self.text_config.filter_duplicate_lines:
            lines = text.split('\n')
            seen_lines = set()
            unique_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and line_stripped not in seen_lines:
                    seen_lines.add(line_stripped)
                    unique_lines.append(line)
            
            text = '\n'.join(unique_lines)
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """
        使用fasttext lid模型检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码 (如 'en', 'zh')
        """
        try:
            import fasttext
            
            # 加载fasttext语言识别模型
            # 实际使用时需要下载lid.176.bin模型
            # model = fasttext.load_model('lid.176.bin')
            # predictions = model.predict(text.replace('\n', ' '))
            # lang = predictions[0][0].replace('__label__', '')
            # return lang
            
            # 简化处理：使用基本启发式规则
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                return 'zh'
            else:
                return 'en'
            
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return 'unknown'