# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import gSC_pb2 as gSC__pb2


class greeterStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.say_hello = channel.unary_unary(
                '/greeter/say_hello',
                request_serializer=gSC__pb2.hello_request.SerializeToString,
                response_deserializer=gSC__pb2.hello_reply.FromString,
                )


class greeterServicer(object):
    """Missing associated documentation comment in .proto file."""

    def say_hello(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_greeterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'say_hello': grpc.unary_unary_rpc_method_handler(
                    servicer.say_hello,
                    request_deserializer=gSC__pb2.hello_request.FromString,
                    response_serializer=gSC__pb2.hello_reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class greeter(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def say_hello(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/greeter/say_hello',
            gSC__pb2.hello_request.SerializeToString,
            gSC__pb2.hello_reply.FromString,
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
                '/stop_server/stop',
                request_serializer=gSC__pb2.stop_request.SerializeToString,
                response_deserializer=gSC__pb2.stop_reply.FromString,
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
                    request_deserializer=gSC__pb2.stop_request.FromString,
                    response_serializer=gSC__pb2.stop_reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'stop_server', rpc_method_handlers)
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
        return grpc.experimental.unary_unary(request, target, '/stop_server/stop',
            gSC__pb2.stop_request.SerializeToString,
            gSC__pb2.stop_reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
