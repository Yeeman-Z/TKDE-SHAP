# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import fed_proto_pb2 as fed__proto__pb2


class GradDescentServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.grad_descent = channel.unary_unary(
                '/rpc_package.GradDescentService/grad_descent',
                request_serializer=fed__proto__pb2.server_request.SerializeToString,
                response_deserializer=fed__proto__pb2.client_reply.FromString,
                )


class GradDescentServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def grad_descent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GradDescentServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'grad_descent': grpc.unary_unary_rpc_method_handler(
                    servicer.grad_descent,
                    request_deserializer=fed__proto__pb2.server_request.FromString,
                    response_serializer=fed__proto__pb2.client_reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rpc_package.GradDescentService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GradDescentService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def grad_descent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rpc_package.GradDescentService/grad_descent',
            fed__proto__pb2.server_request.SerializeToString,
            fed__proto__pb2.client_reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class GetDataSizeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.get_datasize = channel.unary_unary(
                '/rpc_package.GetDataSizeService/get_datasize',
                request_serializer=fed__proto__pb2.datasize_request.SerializeToString,
                response_deserializer=fed__proto__pb2.datasize_reply.FromString,
                )


class GetDataSizeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def get_datasize(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GetDataSizeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'get_datasize': grpc.unary_unary_rpc_method_handler(
                    servicer.get_datasize,
                    request_deserializer=fed__proto__pb2.datasize_request.FromString,
                    response_serializer=fed__proto__pb2.datasize_reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rpc_package.GetDataSizeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GetDataSizeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def get_datasize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rpc_package.GetDataSizeService/get_datasize',
            fed__proto__pb2.datasize_request.SerializeToString,
            fed__proto__pb2.datasize_reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class stop_serverStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.stop = channel.unary_unary(
                '/rpc_package.stop_server/stop',
                request_serializer=fed__proto__pb2.stop_request.SerializeToString,
                response_deserializer=fed__proto__pb2.stop_reply.FromString,
                )


class stop_serverServicer(object):
    """Missing associated documentation comment in .proto file."""

    def stop(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_stop_serverServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'stop': grpc.unary_unary_rpc_method_handler(
                    servicer.stop,
                    request_deserializer=fed__proto__pb2.stop_request.FromString,
                    response_serializer=fed__proto__pb2.stop_reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rpc_package.stop_server', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class stop_server(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def stop(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rpc_package.stop_server/stop',
            fed__proto__pb2.stop_request.SerializeToString,
            fed__proto__pb2.stop_reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
