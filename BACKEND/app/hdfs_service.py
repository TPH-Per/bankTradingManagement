"""
HDFS Service - Simple HDFS operations without Spark dependency

Provides basic HDFS file operations using WebHDFS REST API.
Works with the Docker HDFS setup.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests library not available. Install with: pip install requests")

logger = logging.getLogger(__name__)


class HDFSService:
    """
    Simple HDFS service using WebHDFS REST API.
    
    Features:
    - List files and directories
    - Upload/download files
    - Create/delete directories
    - Get file status and metadata
    - Check HDFS health
    """
    
    def __init__(
        self,
        namenode_host: str = "localhost",
        namenode_port: int = 9870,
        webhdfs_port: int = 9870,
        base_path: str = "/banktrading"
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required. Install with: pip install requests")
        
        self.namenode_host = namenode_host
        self.namenode_port = namenode_port
        self.webhdfs_port = webhdfs_port
        self.base_path = base_path.rstrip('/')
        
        # WebHDFS base URL
        self.webhdfs_url = f"http://{namenode_host}:{webhdfs_port}/webhdfs/v1"
        
        logger.info(f"HDFS Service initialized: {self.webhdfs_url}, base_path={self.base_path}")
    
    def _make_request(
        self,
        method: str,
        hdfs_path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        files: Optional[Dict] = None,
        allow_redirects: bool = False
    ) -> requests.Response:
        """Make WebHDFS REST API request
        
        Args:
            allow_redirects: If False, don't automatically follow redirects (for manual handling)
        """
        # Normalize path: if path starts with '/', use as-is; otherwise prepend base_path
        # Ensure no double slashes
        if hdfs_path.startswith('/'):
            full_path = hdfs_path
        else:
            # Join base_path and hdfs_path, ensuring single slash between them
            base = self.base_path.rstrip('/')
            path = hdfs_path.lstrip('/')
            full_path = f"{base}/{path}" if path else base
        
        url = f"{self.webhdfs_url}{full_path}"
        
        request_params = params or {}
        request_params['op'] = request_params.get('op', 'GETFILESTATUS')
        
        logger.debug(f"HDFS {method} request: {url} with params: {request_params}")
        
        try:
            if method == 'GET':
                response = requests.get(url, params=request_params, timeout=30, allow_redirects=allow_redirects)
            elif method == 'PUT':
                response = requests.put(url, params=request_params, data=data, timeout=60, allow_redirects=allow_redirects)
            elif method == 'POST':
                response = requests.post(url, params=request_params, files=files, timeout=60, allow_redirects=allow_redirects)
            elif method == 'DELETE':
                response = requests.delete(url, params=request_params, timeout=30, allow_redirects=allow_redirects)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to HDFS at {url}: {e}")
            raise ConnectionError(f"HDFS not available at {self.namenode_host}:{self.webhdfs_port}")
        except Exception as e:
            logger.exception(f"Error making HDFS request to {url}: {e}")
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """Check if HDFS is accessible"""
        try:
            # Try to access NameNode Web UI
            response = requests.get(
                f"http://{self.namenode_host}:{self.namenode_port}/jmx",
                timeout=10
            )
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "namenode": f"{self.namenode_host}:{self.namenode_port}",
                    "webhdfs": f"{self.namenode_host}:{self.webhdfs_port}",
                    "base_path": self.base_path
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def list_directory(self, hdfs_path: str = "/") -> List[Dict[str, Any]]:
        """List files and directories in HDFS path"""
        try:
            response = self._make_request('GET', hdfs_path, params={'op': 'LISTSTATUS'})
            response.raise_for_status()
            
            data = response.json()
            files = []
            
            if 'FileStatuses' in data and 'FileStatus' in data['FileStatuses']:
                for item in data['FileStatuses']['FileStatus']:
                    files.append({
                        "path": item.get('pathSuffix', ''),
                        "type": "DIRECTORY" if item.get('type') == 'DIRECTORY' else "FILE",
                        "length": item.get('length', 0),
                        "modificationTime": item.get('modificationTime', 0),
                        "permission": item.get('permission', ''),
                        "owner": item.get('owner', ''),
                        "group": item.get('group', '')
                    })
            
            return files
        except Exception as e:
            logger.exception(f"Error listing directory {hdfs_path}: {e}")
            raise
    
    def get_file_status(self, hdfs_path: str) -> Dict[str, Any]:
        """Get file/directory status"""
        try:
            response = self._make_request('GET', hdfs_path, params={'op': 'GETFILESTATUS'})
            response.raise_for_status()
            
            data = response.json()
            if 'FileStatus' in data:
                status = data['FileStatus']
                return {
                    "path": hdfs_path,
                    "type": "DIRECTORY" if status.get('type') == 'DIRECTORY' else "FILE",
                    "length": status.get('length', 0),
                    "modificationTime": status.get('modificationTime', 0),
                    "permission": status.get('permission', ''),
                    "owner": status.get('owner', ''),
                    "group": status.get('group', ''),
                    "blockSize": status.get('blockSize', 0),
                    "replication": status.get('replication', 0)
                }
            return {}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": "File not found"}
            raise
        except Exception as e:
            logger.exception(f"Error getting file status {hdfs_path}: {e}")
            raise
    
    def create_directory(self, hdfs_path: str) -> Dict[str, Any]:
        """Create directory in HDFS"""
        try:
            response = self._make_request('PUT', hdfs_path, params={'op': 'MKDIRS'})
            response.raise_for_status()
            
            data = response.json()
            return {
                "status": "success",
                "path": hdfs_path,
                "created": data.get('boolean', False)
            }
        except Exception as e:
            logger.exception(f"Error creating directory {hdfs_path}: {e}")
            raise
    
    def upload_file(self, local_file_path: str, hdfs_path: str) -> Dict[str, Any]:
        """Upload local file to HDFS"""
        try:
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")
            
            file_size = os.path.getsize(local_file_path)
            logger.info(f"Uploading local file to HDFS: {local_file_path} -> {hdfs_path} (size: {file_size} bytes)")
            
            # Step 1: Create file (redirect) - disable auto redirect to handle manually
            response = self._make_request('PUT', hdfs_path, params={
                'op': 'CREATE',
                'overwrite': 'true',
                'createparent': 'true'  # Ensure parent directories are created
            }, allow_redirects=False)
            
            logger.debug(f"CREATE response status: {response.status_code}, headers: {dict(response.headers)}")
            
            if response.status_code == 307:  # Temporary redirect
                # Step 2: Upload data to redirect location
                upload_url = response.headers.get('Location')
                if not upload_url:
                    error_msg = f"No redirect location in HDFS response. Status: {response.status_code}, Headers: {dict(response.headers)}"
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                # Fix Docker hostnames in redirect URL
                original_url = upload_url
                upload_url = self._fix_redirect_url(upload_url)
                logger.info(f"Uploading to redirect URL: {upload_url} (original: {original_url})")
                
                try:
                    with open(local_file_path, 'rb') as f:
                        upload_response = requests.put(
                            upload_url, 
                            data=f, 
                            timeout=600, 
                            headers={
                                'Content-Type': 'application/octet-stream',
                                'User-Agent': 'HDFSService/1.0',
                                'Connection': 'close'  # Close connection after request
                            }, 
                            allow_redirects=False
                        )
                    logger.debug(f"Upload response status: {upload_response.status_code}")
                    upload_response.raise_for_status()
                    logger.info(f"Successfully uploaded file to HDFS: {hdfs_path}")
                except requests.exceptions.RequestException as e:
                    error_msg = f"Failed to upload data to DataNode. URL: {upload_url}, Error: {str(e)}"
                    logger.error(error_msg)
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Response status: {e.response.status_code}, Response body: {e.response.text[:500]}")
                    raise ConnectionError(error_msg) from e
                
                return {
                    "status": "success",
                    "path": hdfs_path,
                    "size": file_size,
                    "uploaded_at": datetime.now().isoformat()
                }
            else:
                # Unexpected status code - log details for debugging
                error_msg = f"Unexpected response status {response.status_code} (expected 307 redirect). Response: {response.text[:500]}"
                logger.error(error_msg)
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    raise ConnectionError(f"HDFS CREATE failed with status {response.status_code}: {response.text[:500]}") from e
                # If no exception was raised, it's a success code but not 307 - this shouldn't happen
                logger.warning(f"Received non-307 success code {response.status_code}, treating as success")
                return {"status": "success", "path": hdfs_path}
        except Exception as e:
            logger.exception(f"Error uploading file {local_file_path} to {hdfs_path}: {e}")
            raise
    
    def _fix_redirect_url(self, redirect_url: str) -> str:
        """Fix redirect URL by replacing Docker hostnames with localhost and correcting ports"""
        # Replace Docker hostnames with localhost for host machine access
        if not redirect_url:
            return redirect_url
            
        try:
            from urllib.parse import urlparse, urlunparse
            
            # Parse the URL to handle port replacement properly
            parsed = urlparse(redirect_url)
            
            # Fix hostname: replace datanode/namenode with localhost
            hostname = parsed.hostname or ''
            port = parsed.port
            
            # Handle DataNode redirects
            # WebHDFS uses port 9864 for HTTP uploads (not 9866 which is DataXceiver binary protocol)
            if 'datanode' in hostname.lower() or 'datanode' in redirect_url.lower():
                # Keep port 9864 for WebHDFS HTTP uploads
                # Port 9866 is DataXceiver (binary protocol), not HTTP
                if port == 9866:
                    port = 9864
                    logger.warning(f"Correcting DataNode port from 9866 to 9864 for WebHDFS HTTP upload")
                elif not port or port == 80:
                    # If no port specified or default HTTP, use 9864 for WebHDFS
                    port = 9864
                hostname = 'localhost'
            elif 'namenode' in hostname.lower() or 'namenode' in redirect_url.lower():
                hostname = 'localhost'
                # NameNode WebHDFS uses port 9870
                if not port or port == 80:
                    port = 9870
            
            # Also handle direct localhost URLs - keep 9864 for WebHDFS HTTP
            # Port 9864 is correct for WebHDFS HTTP uploads
            if hostname == 'localhost' and port == 9866:
                port = 9864
                logger.warning(f"Correcting localhost port from 9866 to 9864 for WebHDFS HTTP upload")
            
            # Fix query parameters for DataNode redirect
            # DataNode needs op=CREATE and namenoderpcaddress parameters
            from urllib.parse import parse_qs, urlencode
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            
            # Keep essential parameters for DataNode
            # DataNode requires op=CREATE and namenoderpcaddress
            essential_params = {}
            
            # Keep op=CREATE - DataNode needs it
            if 'op' in query_params:
                essential_params['op'] = query_params['op']
            
            # Fix namenoderpcaddress parameter
            # IMPORTANT: DataNode runs in Docker and needs to use 'namenode:9000' (Docker network name)
            # NOT 'localhost:9000' because from DataNode's perspective, localhost is the container itself
            # We only change the hostname in the URL (datanode -> localhost), but keep namenoderpcaddress as namenode:9000
            if 'namenoderpcaddress' in query_params:
                namenode_addr = query_params['namenoderpcaddress'][0]
                # Keep namenode:9000 as-is - DataNode needs it to connect via Docker network
                # Don't change to localhost:9000 because DataNode can't reach localhost:9000 from inside container
                essential_params['namenoderpcaddress'] = query_params['namenoderpcaddress']
                logger.debug(f"Keeping namenoderpcaddress as {namenode_addr} for DataNode Docker network access")
            
            # Keep other parameters that might be needed (createparent, overwrite, etc.)
            for key in ['createparent', 'overwrite', 'createflag']:
                if key in query_params:
                    essential_params[key] = query_params[key]
            
            # Reconstruct query string with essential parameters
            fixed_query = urlencode(essential_params, doseq=True) if essential_params else ''
            logger.debug(f"Fixed query parameters for DataNode upload")
            
            # Reconstruct the URL with fixed hostname, port, and query
            netloc = f"{hostname}:{port}" if port else hostname
            fixed_url = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                fixed_query,
                parsed.fragment
            ))
            
            logger.info(f"Fixed redirect URL: {fixed_url} (original: {redirect_url})")
            return fixed_url
        except Exception as e:
            # Fallback: simple string replacement if URL parsing fails
            logger.warning(f"URL parsing failed, using simple replacement: {e}")
            import re
            # Replace datanode hostname with localhost, keep port 9864 for WebHDFS HTTP
            # Don't change port 9864 to 9866 - 9864 is correct for WebHDFS HTTP uploads
            fixed = re.sub(r'datanode:9864', 'localhost:9864', redirect_url, flags=re.IGNORECASE)
            fixed = re.sub(r'datanode:(\d+)', r'localhost:\1', fixed, flags=re.IGNORECASE)
            fixed = re.sub(r'namenode:(\d+)', r'localhost:\1', fixed, flags=re.IGNORECASE)
            # Fix any incorrect port 9866 back to 9864 for WebHDFS
            fixed = re.sub(r'localhost:9866/webhdfs', 'localhost:9864/webhdfs', fixed, flags=re.IGNORECASE)
            logger.info(f"Fixed redirect URL (fallback): {fixed} (original: {redirect_url})")
            return fixed
    
    def upload_file_from_bytes(self, file_data: bytes, hdfs_path: str, file_size: Optional[int] = None) -> Dict[str, Any]:
        """Upload file from bytes to HDFS"""
        try:
            # Step 1: Create file (redirect) - disable auto redirect to handle manually
            logger.info(f"Creating file in HDFS: {hdfs_path} (size: {len(file_data)} bytes)")
            response = self._make_request('PUT', hdfs_path, params={
                'op': 'CREATE',
                'overwrite': 'true',
                'createparent': 'true'  # Ensure parent directories are created
            }, allow_redirects=False)
            
            logger.debug(f"CREATE response status: {response.status_code}, headers: {dict(response.headers)}")
            
            if response.status_code == 307:  # Temporary redirect
                # Step 2: Upload data to redirect location
                upload_url = response.headers.get('Location')
                if not upload_url:
                    error_msg = f"No redirect location in HDFS response. Status: {response.status_code}, Headers: {dict(response.headers)}"
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                # Fix Docker hostnames in redirect URL
                original_url = upload_url
                upload_url = self._fix_redirect_url(upload_url)
                logger.info(f"Uploading {len(file_data)} bytes to redirect URL: {upload_url} (original: {original_url})")
                
                # Upload with proper headers and connection handling
                # Use fresh connection (no session) to avoid connection reuse issues
                try:
                    # For WebHDFS DataNode upload, send data directly
                    # The DataNode expects the raw file data in the request body
                    upload_response = requests.put(
                        upload_url, 
                        data=file_data,  # Send raw bytes
                        timeout=600,  # Increase timeout for large files
                        headers={
                            'Content-Type': 'application/octet-stream',
                            'Content-Length': str(len(file_data)),  # Explicit Content-Length
                            'User-Agent': 'HDFSService/1.0',
                            'Connection': 'close'  # Close connection after request to avoid reuse issues
                        },
                        allow_redirects=False
                    )
                    logger.debug(f"Upload response status: {upload_response.status_code}")
                    upload_response.raise_for_status()
                    logger.info(f"Successfully uploaded file to HDFS: {hdfs_path}")
                except requests.exceptions.RequestException as e:
                    error_msg = f"Failed to upload data to DataNode. URL: {upload_url}, Error: {str(e)}"
                    logger.error(error_msg)
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Response status: {e.response.status_code}, Response body: {e.response.text[:500]}")
                    raise ConnectionError(error_msg) from e
                
                size = file_size or len(file_data)
                return {
                    "status": "success",
                    "path": hdfs_path,
                    "size": size,
                    "uploaded_at": datetime.now().isoformat()
                }
            else:
                # Unexpected status code - log details for debugging
                error_msg = f"Unexpected response status {response.status_code} (expected 307 redirect). Response: {response.text[:500]}"
                logger.error(error_msg)
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    raise ConnectionError(f"HDFS CREATE failed with status {response.status_code}: {response.text[:500]}") from e
                # If no exception was raised, it's a success code but not 307 - this shouldn't happen
                logger.warning(f"Received non-307 success code {response.status_code}, treating as success")
                return {"status": "success", "path": hdfs_path}
        except Exception as e:
            logger.exception(f"Error uploading file to {hdfs_path}: {e}")
            raise
    
    def download_file_to_bytes(self, hdfs_path: str) -> bytes:
        """Download file from HDFS as bytes"""
        try:
            # Step 1: Get redirect URL - disable auto redirect to handle manually
            response = self._make_request('GET', hdfs_path, params={'op': 'OPEN'}, allow_redirects=False)
            
            if response.status_code == 307:  # Temporary redirect
                # Step 2: Download from redirect location
                download_url = response.headers.get('Location')
                if not download_url:
                    raise ConnectionError("No redirect location in HDFS response")
                # Fix Docker hostnames in redirect URL
                download_url = self._fix_redirect_url(download_url)
                logger.debug(f"Downloading from fixed URL: {download_url}")
                download_response = requests.get(download_url, timeout=300)
                download_response.raise_for_status()
                return download_response.content
            else:
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.exception(f"Error downloading file {hdfs_path}: {e}")
            raise
    
    def download_file(self, hdfs_path: str, local_file_path: str) -> Dict[str, Any]:
        """Download file from HDFS to local"""
        try:
            # Step 1: Get redirect URL - disable auto redirect to handle manually
            response = self._make_request('GET', hdfs_path, params={'op': 'OPEN'}, allow_redirects=False)
            
            if response.status_code == 307:  # Temporary redirect
                # Step 2: Download from redirect location
                download_url = response.headers.get('Location')
                if not download_url:
                    raise ConnectionError("No redirect location in HDFS response")
                # Fix Docker hostnames in redirect URL
                download_url = self._fix_redirect_url(download_url)
                logger.debug(f"Downloading from fixed URL: {download_url}")
                download_response = requests.get(download_url, timeout=300)
                download_response.raise_for_status()
                
                # Save to local file
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                with open(local_file_path, 'wb') as f:
                    f.write(download_response.content)
                
                file_size = os.path.getsize(local_file_path)
                return {
                    "status": "success",
                    "local_path": local_file_path,
                    "hdfs_path": hdfs_path,
                    "size": file_size,
                    "downloaded_at": datetime.now().isoformat()
                }
            else:
                response.raise_for_status()
                return {"status": "success"}
        except Exception as e:
            logger.exception(f"Error downloading file {hdfs_path}: {e}")
            raise
    
    def delete_file(self, hdfs_path: str, recursive: bool = False) -> Dict[str, Any]:
        """Delete file or directory from HDFS"""
        try:
            params = {'op': 'DELETE'}
            if recursive:
                params['recursive'] = 'true'
            
            response = self._make_request('DELETE', hdfs_path, params=params)
            response.raise_for_status()
            
            data = response.json()
            return {
                "status": "success",
                "path": hdfs_path,
                "deleted": data.get('boolean', False)
            }
        except Exception as e:
            logger.exception(f"Error deleting {hdfs_path}: {e}")
            raise
    
    def get_directory_size(self, hdfs_path: str = "/") -> Dict[str, Any]:
        """Get total size of directory"""
        try:
            response = self._make_request('GET', hdfs_path, params={'op': 'GETCONTENTSUMMARY'})
            response.raise_for_status()
            
            data = response.json()
            if 'ContentSummary' in data:
                summary = data['ContentSummary']
                return {
                    "path": hdfs_path,
                    "directoryCount": summary.get('directoryCount', 0),
                    "fileCount": summary.get('fileCount', 0),
                    "length": summary.get('length', 0),
                    "spaceQuota": summary.get('spaceQuota', -1),
                    "spaceConsumed": summary.get('spaceConsumed', 0)
                }
            return {}
        except Exception as e:
            logger.exception(f"Error getting directory size {hdfs_path}: {e}")
            raise


def create_hdfs_service(
    namenode_host: Optional[str] = None,
    namenode_port: Optional[int] = None,
    base_path: Optional[str] = None
) -> Optional[HDFSService]:
    """
    Factory function to create HDFS service from environment variables.
    Returns None if HDFS is not enabled or not available.
    """
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available. HDFS service cannot be created.")
        return None
    
    namenode_host = namenode_host or os.getenv("HDFS_NAMENODE_HOST", "localhost")
    namenode_port = namenode_port or int(os.getenv("HDFS_NAMENODE_PORT", "9870"))
    base_path = base_path or os.getenv("HDFS_BASE_PATH", "/banktrading")
    
    try:
        service = HDFSService(
            namenode_host=namenode_host,
            namenode_port=namenode_port,
            webhdfs_port=namenode_port,
            base_path=base_path
        )
        # Test connection
        health = service.check_health()
        if health.get("status") != "healthy":
            logger.warning(f"HDFS health check failed: {health}")
            return None
        return service
    except Exception as e:
        logger.warning(f"HDFS service not available: {e}")
        return None

